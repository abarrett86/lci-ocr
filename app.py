# app.py
import os
import io
import gc
import json
import time
import asyncio
import logging
import urllib.request
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import numpy as np
import fitz  # PyMuPDF
from paddleocr import PaddleOCR

# ----------------------------
# Logging setup
# ----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("ocr_app")

try:
    import psutil  # optional; helpful on Heroku to watch RSS
    _PROC = psutil.Process(os.getpid())
except Exception:  # pragma: no cover
    psutil = None
    _PROC = None

def _mem():
    if _PROC is None:
        return None
    try:
        rss = _PROC.memory_info().rss
        return f"{rss/1024/1024:.1f} MB"
    except Exception:
        return None

def _log_mem(prefix: str):
    m = _mem()
    if m:
        log.info("%s | rss=%s", prefix, m)

# ----------------------------
# Config / globals
# ----------------------------
LANG = os.getenv("PADDLEOCR_LANG", "en")
USE_ANGLE_CLS = os.getenv("PADDLEOCR_USE_ANGLE_CLS", "false").lower() == "true"

OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "1600"))
OCR_MAX_PIXELS = int(os.getenv("OCR_MAX_PIXELS", "3000000"))
OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "1"))

# Limit BLAS threads (keeps RAM/CPU stable on small dynos)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

app = FastAPI(title="PaddleOCR REST", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten to your n8n origin if desired
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

OCR_SEMAPHORE = asyncio.Semaphore(OCR_CONCURRENCY)

log.info(
    "Starting app | lang=%s use_angle_cls=%s max_side=%s max_pixels=%s concurrency=%s",
    LANG, USE_ANGLE_CLS, OCR_MAX_SIDE, OCR_MAX_PIXELS, OCR_CONCURRENCY
)
_ = _log_mem("post-init")

# ----------------------------
# Lazy OCR loader (v2.7 friendly, low idle RAM)
# ----------------------------
_OCR: Optional[PaddleOCR] = None

def _get_ocr() -> PaddleOCR:
    global _OCR
    if _OCR is None:
        t0 = time.perf_counter()
        _OCR = PaddleOCR(
            use_angle_cls=USE_ANGLE_CLS,
            lang=LANG,
            use_gpu=False,  # force CPU on Heroku
        )
        log.info("PaddleOCR initialized in %.2fs", time.perf_counter() - t0)
        _log_mem("after PaddleOCR init")
    return _OCR

# ----------------------------
# Helpers
# ----------------------------
def _pil_from_bytes(data: bytes) -> Image.Image:
    log.debug("Entering _pil_from_bytes | bytes=%s", len(data))
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        log.debug("Decoded image | size=%s mode=%s", img.size, img.mode)
        return img
    except Exception as e:
        log.exception("Invalid image data: %s", e)
        raise HTTPException(status_code=400, detail="Invalid image data")

def _choose_page_indices(total_pages: int, page: Optional[int], max_pages: Optional[int]) -> List[int]:
    if page is not None:
        i = page - 1
        if i < 0 or i >= total_pages:
            raise HTTPException(status_code=400, detail=f"PDF has {total_pages} pages; page {page} is out of range")
        return [i]
    idxs = list(range(total_pages))
    if max_pages:
        idxs = idxs[:max(0, int(max_pages))]
    return idxs

def _iter_pdf_pages(pdf_bytes: bytes, dpi: int = 120,
                    page: Optional[int] = None, max_pages: Optional[int] = None):
    """
    Yield (page_index, PIL RGB) one page at a time using PyMuPDF (fitz).
    Grayscale render (smaller) → convert to RGB for OCR.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF data")

    try:
        zoom = (dpi or 120) / 72.0
        mtx = fitz.Matrix(zoom, zoom)
        indices = _choose_page_indices(len(doc), page, max_pages)
        for i in indices:
            pg = doc.load_page(i)
            pm = pg.get_pixmap(matrix=mtx, colorspace=fitz.csGRAY, alpha=False)
            pil = Image.frombytes("L", [pm.width, pm.height], pm.samples).convert("RGB")
            del pm
            gc.collect()
            yield (i, pil)
            del pil
            gc.collect()
    finally:
        doc.close()

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
    log.info("OCR one image | det=%s rec=%s cls=%s size=%s", det, rec, cls, pil.size)
    _log_mem("before OCR")
    t = time.perf_counter()
    pil = _downscale_max_side(pil, OCR_MAX_SIDE)
    pil = _cap_pixels(pil, OCR_MAX_PIXELS)
    arr = np.asarray(pil)[:, :, ::-1]  # RGB -> BGR
    out = _get_ocr().ocr(arr, det=det, rec=rec, cls=cls) or []
    dt = time.perf_counter() - t
    log.info("OCR done in %.2fs | lines=%s", dt, sum(len(line) for line in out))
    items = []
    for line in out:
        for box, (text, score) in line:
            items.append({"bbox": box, "text": text, "score": float(score)})
    del arr, out
    gc.collect()
    _log_mem("after OCR")
    return items

# ----------------------------
# Routes
# ----------------------------
@app.get("/healthz")
def healthz():
    log.debug("healthz called")
    return {
        "ok": True,
        "lang": LANG,
        "use_angle_cls": USE_ANGLE_CLS,
        "max_side": OCR_MAX_SIDE,
        "max_pixels": OCR_MAX_PIXELS,
        "concurrency": OCR_CONCURRENCY,
    }

async def log_request(request: Request):
    log.info("----- Incoming Request -----")
    log.info("URL: %s %s", request.method, request.url)
    log.info("Headers: %s", dict(request.headers))

    ct = (request.headers.get("content-type") or "").lower()

    if "application/json" in ct:
        try:
            body = await request.json()
            log.info("JSON Body: %s", body)
        except Exception as e:
            log.warning("JSON parse failed: %s", e)

    elif "multipart/form-data" in ct:
        try:
            form = await request.form()
            form_dict = {}
            for k, v in form.items():
                if isinstance(v, UploadFile):
                    form_dict[k] = {
                        "filename": v.filename,
                        "content_type": v.content_type,
                        "size": "binary"
                    }
                else:
                    form_dict[k] = v
            log.info("Form Data: %s", form_dict)
        except Exception as e:
            log.warning("Form parse failed: %s", e)

    elif "application/x-www-form-urlencoded" in ct:
        try:
            form = await request.form()
            log.info("Form URL Encoded: %s", dict(form))
        except Exception as e:
            log.warning("Form-URL-Encoded parse failed: %s", e)

    else:
        # For raw uploads (pdf/image)
        body = await request.body()
        log.info("Raw Body Bytes: %s", len(body))

    log.info("----- End Request -----")

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
    await log_request(request)
    ct = (request.headers.get("content-type") or "").lower()
    log.info(
        "POST /ocr | ct=%s det=%s rec=%s cls=%s page=%s max_pages=%s dpi=%s",
        ct, det, rec, cls, page, max_pages, dpi
    )
    _log_mem("enter /ocr")

    # ----- 1) JSON mode (URL) -----
    if image is None and "application/json" in ct:
        log.info("Branch: JSON body")
        try:
            data = await request.json()
            log.debug("JSON keys=%s", list(data.keys()))
        except Exception as e:
            log.exception("Failed to parse JSON: %s", e)
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        url = data.get("url")
        det = data.get("det", det)
        rec = data.get("rec", rec)
        cls = data.get("cls", cls)
        page = data.get("page", page)
        max_pages = data.get("max_pages", max_pages)
        dpi = data.get("dpi", dpi)
        log.info("JSON params | url=%s", url)

    uf: Optional[UploadFile] = image

    # ----- 2) Multipart with arbitrary field name -----
    if uf is None and "multipart/form-data" in ct:
        log.info("Branch: multipart/form-data")
        try:
            form = await request.form()
            log.debug("Form keys=%s", list(form.keys()))
        except Exception as e:
            log.exception("Failed to parse multipart form: %s", e)
            raise HTTPException(status_code=400, detail="Invalid multipart form")
        for key in ("image", "file", "data", "upload"):
            v = form.get(key)
            if isinstance(v, UploadFile):
                uf = v
                log.info("Found UploadFile under key='%s' name='%s' type='%s'", key, v.filename, v.content_type)
                break
        if uf is None:
            for v in form.values():
                if isinstance(v, UploadFile):
                    uf = v
                    log.info("Found UploadFile under unknown key name='%s' type='%s'", v.filename, v.content_type)
                    break

    # ----- 3) Raw body (application/pdf or image/*) -----
    if uf is None and ("application/pdf" in ct or ct.startswith("image/")):
        log.info("Branch: raw body | ct=%s", ct)
        raw = await request.body()
        log.info("Raw bytes=%s", len(raw))
        is_pdf = "application/pdf" in ct
        async with OCR_SEMAPHORE:
            log.debug("Semaphore acquired (raw)")
            try:
                if is_pdf:
                    pages = []
                    for idx, pil in _iter_pdf_pages(raw, dpi=dpi or 120, page=page, max_pages=max_pages):
                        items = _ocr_one_image(pil, det=det, rec=rec, cls=cls)
                        pages.append({"page": idx + 1, "items": items})
                    log.info("Returning %s page(s) [raw]", len(pages))
                    return JSONResponse({"pages": pages})
                else:
                    pil = _pil_from_bytes(raw)
                    items = _ocr_one_image(pil, det=det, rec=rec, cls=cls)
                    log.info("Returning items=%s [raw image]", len(items))
                    return JSONResponse({"items": items})
            finally:
                _log_mem("after raw branch")

    # ----- 4) URL mode (form/json) -----
    if uf is None and url:
        log.info("Branch: URL fetch | url=%s", url)
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                img_bytes = resp.read()
            log.info("Fetched bytes=%s", len(img_bytes))
        except Exception as e:
            log.exception("Failed to fetch image from URL: %s", e)
            raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
        async with OCR_SEMAPHORE:
            log.debug("Semaphore acquired (url)")
            try:
                pil = _pil_from_bytes(img_bytes)
                items = _ocr_one_image(pil, det=det, rec=rec, cls=cls)
                log.info("Returning items=%s [url]", len(items))
                return JSONResponse({"items": items})
            finally:
                _log_mem("after url branch")

    # ----- 5) Multipart UploadFile path -----
    if uf is not None:
        log.info("Branch: multipart UploadFile | name=%s type=%s", uf.filename, uf.content_type)
        file_bytes = await uf.read()
        log.info("UploadFile bytes=%s", len(file_bytes))
        fname = (uf.filename or "").lower()
        is_pdf = ("application/pdf" in (uf.content_type or "").lower()) or fname.endswith(".pdf")
        async with OCR_SEMAPHORE:
            log.debug("Semaphore acquired (multipart)")
            try:
                if is_pdf:
                    pages = []
                    for idx, pil in _iter_pdf_pages(file_bytes, dpi=dpi or 120, page=page, max_pages=max_pages):
                        items = _ocr_one_image(pil, det=det, rec=rec, cls=cls)
                        pages.append({"page": idx + 1, "items": items})
                    log.info("Returning %s page(s) [multipart]", len(pages))
                    return JSONResponse({"pages": pages})
                else:
                    pil = _pil_from_bytes(file_bytes)
                    items = _ocr_one_image(pil, det=det, rec=rec, cls=cls)
                    log.info("Returning items=%s [multipart image]", len(items))
                    return JSONResponse({"items": items})
            finally:
                _log_mem("after multipart branch")

    log.error("No image/PDF provided (multipart, raw, or url expected)")
    raise HTTPException(status_code=400, detail="No image/PDF provided (multipart, raw, or url expected)")