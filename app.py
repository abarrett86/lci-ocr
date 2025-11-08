# app.py
import os
import io
import gc
import json
import asyncio
import urllib.request
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import numpy as np
# --- removed: import fitz  # PyMuPDF
from pdf2image import convert_from_bytes, pdfinfo_from_bytes  # <-- add this
from paddleocr import PaddleOCR


# ----------------------------
# Config / globals
# ----------------------------
LANG = os.getenv("PADDLEOCR_LANG", "en")
USE_ANGLE_CLS = os.getenv("PADDLEOCR_USE_ANGLE_CLS", "false").lower() == "true"

OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "1600"))
OCR_MAX_PIXELS = int(os.getenv("OCR_MAX_PIXELS", "3000000"))

OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "1"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

app = FastAPI(title="PaddleOCR REST", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

OCR_SEMAPHORE = asyncio.Semaphore(OCR_CONCURRENCY)

ocr = PaddleOCR(use_angle_cls=USE_ANGLE_CLS, lang=LANG)


# ----------------------------
# Helpers
# ----------------------------
def _pil_from_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")


def _choose_page_indices(num_pages: int, page: Optional[int], max_pages: Optional[int]) -> List[int]:
    """Determine which 0-based page indices to render."""
    if page is not None:
        i = page - 1
        if i < 0 or i >= num_pages:
            raise HTTPException(status_code=400, detail=f"PDF has {num_pages} pages; page {page} is out of range")
        return [i]
    idxs = list(range(num_pages))
    if max_pages:
        idxs = idxs[:max(0, int(max_pages))]
    return idxs


def _iter_pdf_pages(pdf_bytes: bytes, dpi: int = 120,
                    page: Optional[int] = None, max_pages: Optional[int] = None):
    """
    Yield (page_index, PIL RGB) one page at a time using pdf2image.
    We render each requested page with its own convert_from_bytes call to keep RAM low.
    """
    try:
        info = pdfinfo_from_bytes(pdf_bytes)
        num_pages = int(info.get("Pages", 0))
        if num_pages <= 0:
            raise HTTPException(status_code=400, detail="Invalid or empty PDF")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF data")

    # Select pages (0-based indices)
    indices = _choose_page_indices(num_pages, page, max_pages)

    # pdf2image uses 1-based page numbers for first_page/last_page
    for i in indices:
        imgs = convert_from_bytes(
            pdf_bytes,
            dpi=dpi or 120,
            first_page=i + 1,
            last_page=i + 1,
            fmt="png",           # render via poppler; returns PIL.Image instances
            grayscale=True,      # smaller & fine for OCR; convert to RGB below
            thread_count=1       # be nice if running multiple requests
        )
        if not imgs:
            continue
        pil = imgs[0]
        # ensure RGB for OCR (Paddle expects BGR array later)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        yield (i, pil)
        # cleanup
        del pil, imgs
        gc.collect()


def _downscale_max_side(pil: Image.Image, max_side: int = OCR_MAX_SIDE) -> Image.Image:
    w, h = pil.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return pil


def _cap_pixels(pil: Image.Image, max_pixels: int = OCR_MAX_PIXELS) -> Image.Image:
    w, h = pil.size
    cur = w * h
    if cur <= max_pixels:
        return pil
    scale = (max_pixels / float(cur)) ** 0.5
    pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return pil


def _ocr_one_image(pil: Image.Image, det=True, rec=True, cls=USE_ANGLE_CLS):
    pil = _downscale_max_side(pil, OCR_MAX_SIDE)
    pil = _cap_pixels(pil, OCR_MAX_PIXELS)
    arr = np.asarray(pil)[:, :, ::-1]  # RGB -> BGR
    out = ocr.ocr(arr, det=det, rec=rec, cls=cls) or []
    items = []
    for line in out:
        for box, (text, score) in line:
            items.append({"bbox": box, "text": text, "score": float(score)})
    del arr, out
    gc.collect()
    return items


# ----------------------------
# Routes
# ----------------------------
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "lang": LANG,
        "use_angle_cls": USE_ANGLE_CLS,
        "max_side": OCR_MAX_SIDE,
        "max_pixels": OCR_MAX_PIXELS,
        "concurrency": OCR_CONCURRENCY,
    }


@app.post("/ocr")
async def ocr_endpoint(
    request: Request,
    image: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    det: Optional[bool] = Form(default=True),
    rec: Optional[bool] = Form(default=True),
    cls: Optional[bool] = Form(default=USE_ANGLE_CLS),
    page: Optional[int] = Form(default=None, description="1-based page number"),
    max_pages: Optional[int] = Form(default=None, description="limit how many pages to process"),
    dpi: Optional[int] = Form(default=120, description="PDF rasterization DPI"),
):
    ct = (request.headers.get("content-type") or "").lower()

    # ----- 1) JSON mode (URL) -----
    if image is None and "application/json" in ct:
        data = await request.json()
        url = data.get("url")
        det = data.get("det", det)
        rec = data.get("rec", rec)
        cls = data.get("cls", cls)
        page = data.get("page", page)
        max_pages = data.get("max_pages", max_pages)
        dpi = data.get("dpi", dpi)

    uf: Optional[UploadFile] = image

    # ----- 2) Multipart with arbitrary field name -----
    if uf is None and "multipart/form-data" in ct:
        form = await request.form()
        for key in ("image", "file", "data", "upload"):
            v = form.get(key)
            if isinstance(v, UploadFile):
                uf = v
                break
        if uf is None:
            for v in form.values():
                if isinstance(v, UploadFile):
                    uf = v
                    break

    # ----- 3) Raw body (application/pdf or image/*) -----
    if uf is None and ("application/pdf" in ct or ct.startswith("image/")):
        raw = await request.body()
        is_pdf = "application/pdf" in ct
        async with OCR_SEMAPHORE:
            if is_pdf:
                pages = []
                for idx, pil in _iter_pdf_pages(raw, dpi=dpi or 120, page=page, max_pages=max_pages):
                    items = _ocr_one_image(pil, det=det, rec=rec, cls=cls)
                    pages.append({"page": idx + 1, "items": items})
                return JSONResponse({"pages": pages})
            else:
                pil = _pil_from_bytes(raw)
                items = _ocr_one_image(pil, det=det, rec=rec, cls=cls)
                return JSONResponse({"items": items})

    # ----- 4) URL mode (form/json) -----
    if uf is None and url:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                img_bytes = resp.read()
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
        async with OCR_SEMAPHORE:
            pil = _pil_from_bytes(img_bytes)
            items = _ocr_one_image(pil, det=det, rec=rec, cls=cls)
            return JSONResponse({"items": items})

    # ----- 5) Multipart UploadFile path -----
    if uf is not None:
        file_bytes = await uf.read()
        fname = (uf.filename or "").lower()
        is_pdf = ("application/pdf" in (uf.content_type or "").lower()) or fname.endswith(".pdf")
        async with OCR_SEMAPHORE:
            if is_pdf:
                pages = []
                for idx, pil in _iter_pdf_pages(file_bytes, dpi=dpi or 120, page=page, max_pages=max_pages):
                    items = _ocr_one_image(pil, det=det, rec=rec, cls=cls)
                    pages.append({"page": idx + 1, "items": items})
                return JSONResponse({"pages": pages})
            else:
                pil = _pil_from_bytes(file_bytes)
                items = _ocr_one_image(pil, det=det, rec=rec, cls=cls)
                return JSONResponse({"items": items})

    raise HTTPException(status_code=400, detail="No image/PDF provided (multipart, raw, or url expected)")