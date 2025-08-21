# server.py
import base64
import os
import tempfile
from typing import List, Tuple, Dict, Any, Optional

import requests
import fitz  # PyMuPDF
from paddleocr import PaddleOCR

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import multiprocessing as mp

# ----------------------------
# Input schema (Pydantic)
# ----------------------------
class OCRRequest(BaseModel):
    pdf_url: Optional[str] = Field(default=None, description="Prefer this for big PDFs.")
    pdf_b64: Optional[str] = Field(default=None, description="Base64-encoded PDF.")
    lang: str = Field(default="en", description="PaddleOCR language code")
    dpi: int = Field(default=200, ge=72, le=600, description="Rendering DPI (72-600).")
    page_indices: Optional[List[int]] = Field(
        default=None, description="Zero-based page indices to process; default = all pages."
    )
    join_with: str = Field(default="\n", description="Joiner for combined text output.")

    @field_validator("page_indices")
    @classmethod
    def _non_negative(cls, v):
        if v is not None and any(i < 0 for i in v):
            raise ValueError("page_indices must be >= 0")
        return v

    @field_validator("pdf_b64", "pdf_url")
    @classmethod
    def _empty_to_none(cls, v):
        return v or None

    @field_validator("*")
    @classmethod
    def _require_one_of(cls, v, info):
        # This runs for each field; only check once at the end
        if info.field_name != "join_with":
            return v
        values = info.data
        if not values.get("pdf_url") and not values.get("pdf_b64"):
            raise ValueError("You must provide either pdf_url or pdf_b64.")
        return v


# ----------------------------
# OCR engine cache (per-lang, per worker process)
# ----------------------------
_OCR_ENGINES: Dict[str, PaddleOCR] = {}

def get_ocr(lang: str) -> PaddleOCR:
    """Create/cache a PaddleOCR engine per language (within a worker process)."""
    if lang not in _OCR_ENGINES:
        _OCR_ENGINES[lang] = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return _OCR_ENGINES[lang]


# ----------------------------
# PDF loading helpers
# ----------------------------
def _load_pdf_bytes(job_input: Dict[str, Any]) -> bytes:
    if job_input.get("pdf_url"):
        resp = requests.get(job_input["pdf_url"], timeout=60)
        resp.raise_for_status()
        return resp.content

    b64 = job_input.get("pdf_b64", "")
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    return base64.b64decode(b64)


# ----------------------------
# Page -> OCR text
# ----------------------------
def _render_page_to_png_bytes(doc: fitz.Document, page_index: int, dpi: int) -> bytes:
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
    return pix.tobytes("png")


def _ocr_png_bytes(png_bytes: bytes, ocr: PaddleOCR) -> List[Tuple[float, float, str, float]]:
    """
    Run OCR on a PNG image (bytes). Returns list of (y, x, text, score)
    where (x,y) is the centroid of the detected polygon (used for sorting).
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes)
        tmp_path = tmp.name

    try:
        results = ocr.predict(tmp_path)  # PaddleOCR v3.x returns list of Result objects
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    lines: List[Tuple[float, float, str, float]] = []
    if not results:
        return lines

    for res in results:
        try:
            d = getattr(res, "res", None) or {}
            texts = d.get("rec_texts") or []
            scores = d.get("rec_scores") or []
            polys = d.get("rec_polys") or d.get("dt_polys") or []

            n = min(len(texts), len(scores), len(polys))
            for i in range(n):
                text = texts[i]
                score = float(scores[i])
                poly = polys[i]
                try:
                    cx = sum(p[0] for p in poly) / max(len(poly), 1)
                    cy = sum(p[1] for p in poly) / max(len(poly), 1)
                except Exception:
                    if isinstance(poly, (list, tuple)) and len(poly) >= 8:
                        xs = poly[0::2]
                        ys = poly[1::2]
                        cx = sum(xs) / len(xs)
                        cy = sum(ys) / len(ys)
                    else:
                        cx = 0.0
                        cy = 0.0
                lines.append((cy, cx, text, score))
        except Exception:
            continue

    # reading order: top-to-bottom, then left-to-right
    lines.sort(key=lambda t: (round(t[0], 1), t[1]))
    return lines


def _ocr_pdf(pdf_bytes: bytes, lang: str, dpi: int, page_indices: Optional[List[int]]) -> Dict[str, Any]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count

    if page_indices:
        indices = [i for i in page_indices if 0 <= i < total_pages]
        if not indices:
            raise ValueError("No valid page indices supplied.")
    else:
        indices = list(range(total_pages))

    ocr = get_ocr(lang)
    page_texts: List[str] = []

    for i in indices:
        png = _render_page_to_png_bytes(doc, i, dpi)
        lines = _ocr_png_bytes(png, ocr)
        page_texts.append("\n".join([line[2] for line in lines]))

    return {
        "pages": page_texts,
        "page_count": total_pages,
        "processed_pages": len(indices),
        "processed_page_indices": indices,
        "lang": lang,
        "dpi": dpi,
    }


# ----------------------------
# Worker entry (runs inside Pool)
# ----------------------------
def _run_ocr_job(job_input: Dict[str, Any]) -> Dict[str, Any]:
    try:
        pdf_bytes = _load_pdf_bytes(job_input)
    except Exception as e:
        return {"error": f"Failed to load PDF: {e.__class__.__name__}", "details": str(e)}

    lang = job_input.get("lang", "en")
    dpi = int(job_input.get("dpi", 200))
    page_indices = job_input.get("page_indices")
    join_with = job_input.get("join_with", "\n")

    try:
        ocr_result = _ocr_pdf(pdf_bytes, lang=lang, dpi=dpi, page_indices=page_indices)
        combined = join_with.join(ocr_result["pages"])
        return {
            "text": combined,
            "pages": ocr_result["pages"],
            "meta": {
                "page_count": ocr_result["page_count"],
                "processed_pages": ocr_result["processed_pages"],
                "processed_page_indices": ocr_result["processed_page_indices"],
                "lang": ocr_result["lang"],
                "dpi": ocr_result["dpi"],
            },
        }
    except Exception as e:
        return {"error": f"OCR failed: {e.__class__.__name__}", "details": str(e)}


# ----------------------------
# FastAPI app + multiprocessing pool
# ----------------------------
app = FastAPI(title="PDF OCR Server", version="1.0.0")

# We'll create the pool at startup so workers can keep OCR models cached.
POOL: Optional[mp.pool.Pool] = None

@app.on_event("startup")
def _startup():
    global POOL
    # Use "spawn" for cross-platform safety
    ctx = mp.get_context("spawn")
    workers = int(os.getenv("OCR_WORKERS", max(1, (os.cpu_count() or 2) // 2)))
    POOL = ctx.Pool(processes=workers)

@app.on_event("shutdown")
def _shutdown():
    global POOL
    if POOL is not None:
        POOL.close()
        POOL.join()
        POOL = None

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/ocr")
def ocr_endpoint(req: OCRRequest):
    if POOL is None:
        raise HTTPException(status_code=503, detail="Worker pool not initialized.")
    # Run in worker process; keep the web thread light
    async_result = POOL.apply_async(_run_ocr_job, (req.model_dump(),))
    try:
        # Optional: let users override timeout via env
        timeout_s = float(os.getenv("OCR_TIMEOUT_SECONDS", "600"))
        result = async_result.get(timeout=timeout_s)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Worker error: {e}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result)

    return result


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    # Launch with: python server.py
    # Or: uvicorn server:app --host 0.0.0.0 --port 8000
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
