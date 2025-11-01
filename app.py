# app.py
import io, os, json, urllib.request
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

LANG = os.getenv("PADDLEOCR_LANG", "en")
USE_ANGLE_CLS = os.getenv("PADDLEOCR_USE_ANGLE_CLS", "true").lower() == "true"

# Load OCR models once at startup
ocr = PaddleOCR(use_angle_cls=USE_ANGLE_CLS, lang=LANG)

app = FastAPI(title="PaddleOCR REST", version="1.0.0")

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

@app.post("/ocr")
async def ocr_endpoint(
    request: Request,
    image: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    det: Optional[bool] = Form(default=True),
    rec: Optional[bool] = Form(default=True),
    cls: Optional[bool] = Form(default=USE_ANGLE_CLS),
):
    # Allow JSON payload ({"url": ...}) when not multipart
    if image is None and url is None and request.headers.get("content-type","").startswith("application/json"):
        data = await request.json()
        url = data.get("url")
        det = data.get("det", det)
        rec = data.get("rec", rec)
        cls = data.get("cls", cls)

    if image is None and url is None:
        raise HTTPException(status_code=400, detail="Provide image file or url")

    if image is not None:
        img_bytes = await image.read()
    else:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                img_bytes = resp.read()
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to fetch image from URL")

    pil = pil_from_bytes(img_bytes)
    items = run_ocr(pil, det=det, rec=rec, cls=cls)
    return JSONResponse({"items": items})