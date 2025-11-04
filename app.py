# app.py
import io, os, json, urllib.request
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
from fastapi.middleware.cors import CORSMiddleware


import fitz  # PyMuPDF

LANG = os.getenv("PADDLEOCR_LANG", "en")
USE_ANGLE_CLS = os.getenv("PADDLEOCR_USE_ANGLE_CLS", "true").lower() == "true"

# Load OCR models once at startup
ocr = PaddleOCR(use_angle_cls=USE_ANGLE_CLS, lang=LANG)

app = FastAPI(title="PaddleOCR REST", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your n8n URL, e.g. "https://n8n.yourdomain.com"
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

class UrlPayload(BaseModel):
    url: str
    det: Optional[bool] = True
    rec: Optional[bool] = True
    cls: Optional[bool] = USE_ANGLE_CLS

@app.get("/healthz")
def healthz():
    return {"ok": True, "lang": LANG}

def pil_from_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

def run_ocr(img: Image.Image, det=True, rec=True, cls=USE_ANGLE_CLS):
    # PaddleOCR expects BGR ndarray
    arr = np.array(img)[:, :, ::-1]
    result = ocr.ocr(arr, det=det, rec=rec, cls=cls)
    items = []
    if result:
        for line in result:
            for box, (text, score) in line:
                items.append({"bbox": box, "text": text, "score": float(score)})
    return items

def _pil_from_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

def _pil_list_from_pdf_bytes(pdf_bytes: bytes, dpi: int = 180, page: Optional[int] = None, max_pages: Optional[int] = None) -> List[Image.Image]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF data")

    images: List[Image.Image] = []

    # Decide which pages to render
    if page is not None:
        # 1-based -> 0-based
        i = page - 1
        if i < 0 or i >= len(doc):
            raise HTTPException(status_code=400, detail=f"PDF has {len(doc)} pages; page {page} is out of range")
        page_indices = [i]
    else:
        page_indices = list(range(len(doc)))
        if max_pages:
            page_indices = page_indices[:max_pages]

    # Render pages to PIL Images
    zoom = dpi / 72.0
    mtx = fitz.Matrix(zoom, zoom)
    for i in page_indices:
        pg = doc.load_page(i)
        pm = pg.get_pixmap(matrix=mtx, alpha=False)   # RGB
        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
        images.append(img)

    return images

def _run_ocr(img: Image.Image, det=True, rec=True, cls=USE_ANGLE_CLS):
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
    result = ocr.ocr(arr, det=det, rec=rec, cls=cls)
    items = []
    if result:
        for line in result:
            for box, (text, score) in line:
                items.append({"bbox": box, "text": text, "score": float(score)})
    return items

@app.post("/ocr")
async def ocr_endpoint(
    request: Request,
    # keep the explicit field for normal clients
    image: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    det: Optional[bool] = Form(default=True),
    rec: Optional[bool] = Form(default=True),
    cls: Optional[bool] = Form(default=USE_ANGLE_CLS),
    page: Optional[int] = Form(default=None),
    max_pages: Optional[int] = Form(default=None),
    dpi: Optional[int] = Form(default=180),
):
    ct = (request.headers.get("content-type") or "").lower()

    # 1) JSON mode (URL)
    if image is None and "application/json" in ct:
        data = await request.json()
        url = data.get("url")
        det = data.get("det", det)
        rec = data.get("rec", rec)
        cls = data.get("cls", cls)
        page = data.get("page", page)
        max_pages = data.get("max_pages", max_pages)
        dpi = data.get("dpi", dpi)

    # 2) Multipart with arbitrary field name (e.g., "data" in n8n)
    uf: Optional[UploadFile] = image
    if uf is None and "multipart/form-data" in ct:
        form = await request.form()
        # try common names first
        for key in ("image", "file", "data", "upload"):
            if key in form and isinstance(form[key], UploadFile):
                uf = form[key]  # type: ignore[assignment]
                break
        # else just take the first UploadFile in the form
        if uf is None:
            for v in form.values():
                if isinstance(v, UploadFile):
                    uf = v
                    break

    # 3) Raw body (application/pdf or image/*)
    if uf is None and ("application/pdf" in ct or ct.startswith("image/")):
        raw = await request.body()
        is_pdf = "application/pdf" in ct
        if is_pdf:
            pil_pages = _pil_list_from_pdf_bytes(raw, dpi=dpi or 180, page=page, max_pages=max_pages)
            pages = []
            for idx, pil in enumerate(pil_pages, start=1 if page is None else page):
                pages.append({"page": idx, "items": _run_ocr(pil, det=det, rec=rec, cls=cls)})
            return JSONResponse({"pages": pages})
        else:
            pil = _pil_from_bytes(raw)
            return JSONResponse({"items": _run_ocr(pil, det=det, rec=rec, cls=cls)})

    # 4) URL mode if provided via form
    if uf is None and url:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                img_bytes = resp.read()
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
        pil = _pil_from_bytes(img_bytes)
        return JSONResponse({"items": _run_ocr(pil, det=det, rec=rec, cls=cls)})

    # 5) Handle the UploadFile (multipart)
    if uf is not None:
        file_bytes = await uf.read()
        fname = (uf.filename or "").lower()
        is_pdf = ("application/pdf" in (uf.content_type or "").lower()) or fname.endswith(".pdf")
        if is_pdf:
            pil_pages = _pil_list_from_pdf_bytes(file_bytes, dpi=dpi or 180, page=page, max_pages=max_pages)
            pages = []
            for idx, pil in enumerate(pil_pages, start=1 if page is None else page):
                pages.append({"page": idx, "items": _run_ocr(pil, det=det, rec=rec, cls=cls)})
            return JSONResponse({"pages": pages})
        else:
            pil = _pil_from_bytes(file_bytes)
            return JSONResponse({"items": _run_ocr(pil, det=det, rec=rec, cls=cls)})

    raise HTTPException(status_code=400, detail="No image/PDF provided (multipart, raw, or url expected)")