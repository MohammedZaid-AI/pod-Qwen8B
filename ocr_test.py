# ============================================================
# POD OCR TEST
# PaddleOCR + Image Preprocessing + OpenRouter Qwen3-VL-8B
# Windows CPU (OCR) / OpenRouter (vision classification)
# ============================================================

import os
import json
import re
import requests
import cv2
import numpy as np

import re
from datetime import datetime


# Month abbreviations used when parsing OCR dates
_MONTH_ABBR = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


# Matches dates such as:
# 02-JUL-26
# 02/JUL/2026
# 02.JUL.2026
_OCR_TEXT_DATE = re.compile(
    r"\b(\d{1,2})[-/.]([A-Za-z]{3,9})[-/.](\d{2,4})\b",
    re.IGNORECASE
)


# Matches numeric dates such as:
# 02-07-2026
# 02/07/2026
# 02.07.2026
_OCR_NUMERIC_DATE = re.compile(
    r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b"
)

# ============================================================
# PADDLE CONFIGURATION
# ============================================================

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"


# ============================================================
# IMPORTS
# ============================================================

import paddle

from PIL import Image, ImageEnhance, ImageOps
from paddleocr import PaddleOCR
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

print("API key loaded:", bool(api_key))
print("Model:", os.getenv("OPENROUTER_MODEL"))

# Explicitly disable MKLDNN
paddle.set_flags({"FLAGS_use_mkldnn": False})


# ============================================================
# IMAGE PREPROCESSING  (unchanged from your version)
# ============================================================

def _order_document_points(points):
    """Return corners in TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(sums)],
        pts[np.argmin(diffs)],
        pts[np.argmax(sums)],
        pts[np.argmax(diffs)],
    ], dtype=np.float32)


def _find_document_corners(image):
    h, w = image.shape[:2]
    scale = min(1400.0 / max(h, w), 1.0)
    small = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    sh, sw = small.shape[:2]
    image_area = sh * sw

    candidates = []

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 25, 90)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = cv2.contourArea(contour)
        if not image_area * 0.18 < area < image_area * 0.97:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append((area, approx.reshape(4, 2).astype(np.float32)))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    paper_mask = cv2.inRange(
        cv2.merge([hsv[:, :, 0], saturation, value]),
        np.array([0, 0, 115], dtype=np.uint8),
        np.array([179, 105, 255], dtype=np.uint8)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = cv2.contourArea(contour)
        if not image_area * 0.18 < area < image_area * 0.97:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append((area * 1.05, approx.reshape(4, 2).astype(np.float32)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    for area, points in candidates:
        if area < image_area * 0.22 or area > image_area * 0.96:
            continue
        points = points / scale
        ordered = _order_document_points(points)
        tl, tr, br, bl = ordered
        widths = [np.linalg.norm(tr - tl), np.linalg.norm(br - bl)]
        heights = [np.linalg.norm(bl - tl), np.linalg.norm(br - tr)]
        if min(widths) < w * 0.30 or min(heights) < h * 0.30:
            continue
        ratio = max(widths) / max(1.0, max(heights))
        if ratio < 0.25 or ratio > 4.0:
            continue
        return points.astype(np.float32)

    return None


def _perspective_correct(image, corners):
    tl, tr, br, bl = _order_document_points(corners)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 100 or height < 100:
        return image
    destination = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(
        np.array([tl, tr, br, bl], dtype=np.float32), destination
    )
    return cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def _deskew_small_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=100,
                            minLineLength=max(100, min(image.shape[:2]) // 4),
                            maxLineGap=20)
    if lines is None:
        return image
    angles = []
    try:
        line_segments = np.asarray(lines).reshape(-1, 4)
    except (TypeError, ValueError):
        print("WARNING: Unexpected Hough line format; skipping deskew.")
        return image

    for line in line_segments:
        x1, y1, x2, y2 = (float(value) for value in line)
        if x1 == x2 and y1 == y2:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = ((angle + 45) % 90) - 45
        if abs(angle) <= 8:
            angles.append(angle)
    if len(angles) < 3:
        return image
    angle = float(np.median(angles))
    if abs(angle) < 0.35 or abs(angle) > 8:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _digital_scan_effect(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=3, templateWindowSize=7, searchWindowSize=21)
    low, high = np.percentile(gray, (1.0, 99.0))
    if high > low + 5:
        stretched = (gray.astype(np.float32) - low) * (255.0 / (high - low))
        gray = np.clip(stretched, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.15, tileGridSize=(12, 12))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 0.8)
    result = cv2.addWeighted(gray, 1.12, blur, -0.12, 0)
    return result


def preprocess_image(image_path):
    print("\n========== DOCUMENT PREPROCESSING ==========")
    os.makedirs("processed_images", exist_ok=True)

    pil_image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    rgb = np.array(pil_image)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    print(f"Original image size: {pil_image.size}")

    # FIX: "auto" (the default) no longer rotates here at all -- large-angle
    # orientation is now detected and corrected by PaddleOCR's built-in
    # classifier when run_ocr() is called later (see _get_ocr_engine). A
    # manual override, if explicitly set, is still applied immediately so
    # both OCR and the image sent to the vision model start out consistent.
    if POD_ROTATION_SETTING != "auto":
        try:
            forced_angle = int(POD_ROTATION_SETTING) % 360
        except ValueError as exc:
            raise ValueError("POD_ROTATION_DEGREES must be 'auto', 0, 90, 180, or 270") from exc
        if forced_angle not in (0, 90, 180, 270):
            raise ValueError("POD_ROTATION_DEGREES must be 'auto', 0, 90, 180, or 270")
        if forced_angle:
            image = _rotate_candidate(image, forced_angle)
        print(f"Applied manual rotation override: {forced_angle} degrees.")

    corners = _find_document_corners(image)
    if corners is not None:
        document = _perspective_correct(image, corners)
        print("Document boundary detected; perspective correction applied.")
    else:
        document = image.copy()
        print("WARNING: Page boundary ambiguous; preserving full frame (no risky crop).")

    document = _deskew_small_angle(document)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    document_path = os.path.join("processed_images", f"{stem}_document.jpg")
    enhanced_path = os.path.join("processed_images", f"{stem}_ocr_enhanced.png")

    cv2.imwrite(document_path, document, [cv2.IMWRITE_JPEG_QUALITY, 96])
    enhanced = _digital_scan_effect(document)
    cv2.imwrite(enhanced_path, enhanced)

    print(f"Rectified document saved: {document_path}")
    print(f"OCR-enhanced image saved: {enhanced_path}")

    return {
        "document": document_path,
        "enhanced": enhanced_path,
        "crops": {},
        "boundary_detected": corners is not None,
    }


# ============================================================
# OCR MODEL
# ============================================================

_OCR_ENGINE = None

def _get_ocr_engine():
    """
    FIX: enables PaddleOCR's own built-in document-orientation classifier
    (use_doc_orientation_classify) and per-line orientation classifier
    (use_textline_orientation). These are small, dedicated models that run
    as part of the normal .predict() call -- they detect and correct
    0/90/180/270 rotation automatically, in ONE OCR pass.

    This replaces the previous approach of calling .predict() four separate
    times (once per candidate rotation) just to guess orientation with a
    hand-rolled confidence heuristic. That was 5 total OCR calls per image;
    this is 1. It is also more reliable, since it is a purpose-built
    classifier rather than a heuristic based on OCR confidence/keyword
    counting.

    use_doc_unwarping is left off because you already do your own
    perspective correction and deskewing (_find_document_corners,
    _perspective_correct, _deskew_small_angle) -- enabling both would be
    redundant and could interact unpredictably.
    """
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        print("\n========== LOADING OCR MODEL ==========\n")
        _OCR_ENGINE = PaddleOCR(
            lang="en",
            device="cpu",
            use_doc_orientation_classify=True,
            use_textline_orientation=True,
            use_doc_unwarping=False,
        )
    return _OCR_ENGINE

def run_ocr(image_path):
    print("\n========== RUNNING OCR ==========\n")
    return _get_ocr_engine().predict(image_path)

def _rotate_candidate(image, degrees):
    if degrees == 0:
        return image.copy()
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}[degrees]
    return cv2.rotate(image, code)

def extract_orientation_angle(ocr_result):
    """
    Reads the page-rotation angle detected by PaddleOCR's built-in
    document-orientation classifier (enabled in _get_ocr_engine above).
    Returns one of 0/90/180/270 on a confident detection, or None if the
    classifier didn't produce one of those (PaddleOCR commonly reports -1
    for "not confident" or when the module is disabled).

    NOTE: the exact JSON key ("doc_preprocessor_res" -> "angle") matches
    PaddleOCR 3.x's documented pipeline output as of this writing. If your
    installed version differs, run this once and print(data) inside the
    loop below to confirm the real key path, then adjust here -- it's a
    one-line change once you see the actual structure.
    """
    for page in ocr_result:
        if not hasattr(page, "json"):
            continue
        data = page.json
        if callable(data):
            data = data()
        res = data.get("res", {})
        doc_pre = res.get("doc_preprocessor_res") or {}
        angle = doc_pre.get("angle")
        if angle in (0, 90, 180, 270):
            return angle
        if angle is not None:
            print(f"WARNING: unrecognized orientation angle from PaddleOCR: {angle!r}")
        return None
    return None


def extract_ocr_data(result):
    all_data = []
    for page in result:
        if not hasattr(page, "json"):
            continue
        data = page.json
        if callable(data):
            data = data()
        res = data.get("res", {})
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("rec_polys", [])
        for text, score, box in zip(texts, scores, boxes):
            text = str(text).strip()
            if not text or score < 0.50:
                continue
            all_data.append({
                "text": text,
                "confidence": float(score),
                "box": box.tolist() if hasattr(box, "tolist") else box,
            })
    return all_data


def normalize_ocr_data(ocr_data):
    normalized = []
    for item in ocr_data:
        box = item["box"]
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        width, height = x2 - x1, y2 - y1
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        normalized.append({
            "text": item["text"],
            "confidence": round(item["confidence"], 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "center_x": round(center_x, 2), "center_y": round(center_y, 2),
            "width": width, "height": height,
        })
    return normalized


def display_normalized_ocr(ocr_data):
    print("\n========== NORMALIZED OCR DATA ==========\n")
    print(f"{'TEXT':<40} {'SCORE':<8} {'CENTER':<20} {'SIZE':<15}")
    print("-" * 100)
    for item in ocr_data:
        print(
            f"{item['text'][:38]:<40} "
            f"{item['confidence']:<8.3f} "
            f"({item['center_x']:.0f}, {item['center_y']:.0f})"
            f"{'':<7}"
            f"{item['width']:.0f} x {item['height']:.0f}"
        )
    print("\n==========================================\n")


def build_ocr_text(ocr_data):
    lines = []
    for item in ocr_data:
        text = item["text"].strip()
        if not text:
            continue
        lines.append(
            f"text={text!r}; center=({item['center_x']:.0f},{item['center_y']:.0f}); "
            f"confidence={item['confidence']:.3f}"
        )
    return "\n".join(lines)


# ============================================================
# OPENROUTER QWEN3-VL: IMAGE + OCR -> STRUCTURED EVIDENCE
# ============================================================

import base64
from datetime import datetime
from mimetypes import guess_type

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "").strip()
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions").strip()
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "POD Classification").strip()
IMAGE_PATH = os.getenv("POD_IMAGE_PATH", "images/pod4.jpg").strip()
POD_ROTATION_SETTING = os.getenv("POD_ROTATION_DEGREES", "auto").strip().lower()
# Set POD_ROTATION_DEGREES=auto (recommended) or a fixed 0/90/180/270 override.

EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "cnNumber": {"type": ["string", "null"]},
        "deliveryDate": {"type": ["string", "null"]},
        "remarksText": {"type": ["string", "null"]},
        "hasSignature": {"type": "boolean"},
        "hasStamp": {"type": "boolean"},
        "hasHandwriting": {"type": "boolean"},
        "imageQualityPassed": {"type": "boolean"},
        "physicalDamage": {"type": "boolean"},
        "businessDamage": {"type": "boolean"},
        "shortage": {"type": "boolean"},
        "manualReviewRequired": {"type": "boolean"},
        "confidenceScore": {"type": "number"},
        "evidence": {
            "type": "object",
            "properties": {
                "cnNumber": {"type": "string"},
                "deliveryDate": {"type": "string"},
                "remarksText": {"type": "string"},
                "signature": {"type": "string"},
                "stamp": {"type": "string"},
                "physicalDamage": {"type": "string"},
                "businessDamage": {"type": "string"},
                "shortage": {"type": "string"},
                "manualReview": {"type": "string"},
                "imageQuality": {"type": "string"}
            },
            "required": [
                "cnNumber", "deliveryDate", "remarksText", "signature",
                "stamp", "physicalDamage", "businessDamage", "shortage",
                "manualReview", "imageQuality"
            ],
            "additionalProperties": False
        }
    },
    "required": [
        "cnNumber", "deliveryDate", "remarksText", "hasSignature",
        "hasStamp", "hasHandwriting", "imageQualityPassed",
        "physicalDamage", "businessDamage", "shortage",
        "manualReviewRequired", "confidenceScore", "evidence"
    ],
    "additionalProperties": False
}


def validate_configuration():
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")
    if not OPENROUTER_MODEL:
        raise RuntimeError("OPENROUTER_MODEL is missing. Set the exact OpenRouter model ID in .env.")
    if not OPENROUTER_URL.startswith("https://"):
        raise RuntimeError("OPENROUTER_URL must use HTTPS.")


def image_data_url(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime_type, _ = guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_vision_prompt(ocr_text):
    return f"""
You are a careful POD (Proof of Delivery) document evidence extractor.
Inspect the full supplied image. OCR text is supporting evidence only; it may
contain mistakes, printed labels, and text from overlapping sheets.

Return only JSON matching the supplied schema. Do not add markdown.

IMPORTANT: EVALUATE EACH FIELD INDEPENDENTLY
Do not let uncertainty about one field make you ignore other visible fields.
Inspect the entire page, including the top header, barcode area, all signature
areas, stamps/seals, remarks, checkboxes, dates, and lower tables.

CN NUMBER
- Return only a number visually associated with CN, consignment, or docket.
- Prefer a clearly labeled docket/CN number. Do not use GST, phone, PIN,
  invoice, e-waybill, item, or manual-reference numbers.
- If multiple plausible values remain, return null and explain why.

SIGNATURE
- Inspect the full document, especially consignee/receiver signature fields.
- A visible handwritten signature in a relevant signature field counts even
  when OCR cannot read it. A printed name or printed label alone does not.
- Do not mistake unrelated handwriting, initials, or a watermark for the
  delivery signature. If ambiguous, describe the ambiguity and request review.

STAMP / SEAL
- Inspect the whole page for recognizable company/official stamps, including
  colored, diagonal, faint, or text-overlapping stamps.
- Do not require OCR recognition. Do not treat a logo, watermark, or random
  pen stroke as a stamp. If uncertain, request review.

HANDWRITING
- True for visible handwritten content anywhere, including signatures,
  remarks, dates, or annotations. Handwriting does not automatically mean
  signature.

REMARKS AND ISSUE CHECKBOXES
- Locate the actual REMARKS field and inspect it visually. Report entered
  handwritten/typed content, not merely printed prompts or labels.
- Preserve every readable entered remark in remarksText. Do not omit words
  such as "Short" or "Damage" if they are genuinely written/selected as an
  entry; do not copy those words merely because they are printed labels.
- Inspect checkbox interiors separately. A printed word beside an empty box
  is not a selected issue. A clear tick/cross/pen mark inside the checkbox is
  evidence of selection. If a checkbox is tiny, covered, or unclear, say so
  and set manualReviewRequired=true.
- businessDamage means damage to goods/consignment, established by a clear
  selected Damage checkbox or actual entered remarks. shortage means short
  quantity, established by a clear selected Short checkbox or actual entered
  remarks. Do not infer either from a printed label alone.
- If OCR text contains issue words, check whether they are actual entered
  content or only printed labels. Explain the distinction in evidence.

PHYSICAL PAPER DAMAGE
- physicalDamage=true only if the document paper itself is torn, ripped,
  missing a physical section, destroyed, or severely damaged.
- Ordinary folds, wrinkles, shadows, and minor bends do not count.
- This is distinct from damage to delivered goods.

DATES
- deliveryDate must be explicitly associated with delivery/receipt/POD
  delivery. Do not substitute booking, invoice, pickup, shipping, print, MR,
  or unrelated dates.
- Return YYYY-MM-DD only when the delivery date and year are legible and
  unambiguous. Otherwise null. Never complete a partial date by guessing.

IMAGE QUALITY / OVERLAP
- imageQualityPassed=false if blur, darkness, overexposure, cropping, low
  resolution, or overlapping sheets prevent reliable inspection of material
  fields. Some readable OCR does not automatically mean quality passed.
- If a second sheet overlaps the POD or hides fields, mention exactly what is
  obscured and set manualReviewRequired=true when it affects decisions.
- Do not claim overlap unless visible in the image.

EVIDENCE FIELDS
- Every evidence.* field must contain a short, concrete visual observation.
- Never use only "True", "False", "Yes", or "No" as evidence.
- Quote visible text where possible and identify its location (e.g. "remarks
  box at lower right contains handwritten 'Damage'").
- If evidence is uncertain, state the uncertainty rather than inventing proof.

MANUAL REVIEW
Set manualReviewRequired=true if important fields are obscured/ambiguous,
the signature or stamp cannot be determined, the image is poor, or the CN is
ambiguous. Missing date alone need not require review if otherwise reliable.
Do not classify; Python applies the category priority.

OCR TEXT (may be imperfect and may include printed labels):
{ocr_text}
"""


def _extract_json_from_response(response_json):
    choices = response_json.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {response_json}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenRouter returned empty/non-text model content.")
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model response was not valid JSON: {content[:1500]}") from exc


def run_vision_model(image_path, ocr_text):
    validate_configuration()
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": OPENROUTER_APP_NAME,
    }
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL

    image_content = [
        {"type": "text", "text": build_vision_prompt(ocr_text)},
        {"type": "text", "text": "IMAGE 1 is the complete POD page. The full page must remain in view."},
        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
    ]

    payload = {
        "model": OPENROUTER_MODEL,
        "temperature": 0,
        "max_tokens": 1800,
        "messages": [{"role": "user", "content": image_content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "pod_evidence", "strict": True, "schema": EVIDENCE_SCHEMA}
        }
    }

    print("\n========== OPENROUTER VISION MODEL ==========")
    print(f"Model: {OPENROUTER_MODEL}")
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=(20, 240))
        if not response.ok:
            raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {response.text[:2000]}")
        response_json = response.json()
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("OpenRouter request timed out. Try again or reduce image size.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not connect to OpenRouter: {exc}") from exc

    evidence = _extract_json_from_response(response_json)
    validate_evidence(evidence)
    return evidence


def reconcile_cn_with_ocr(evidence, ocr_data):
    """
    Keep the model's CN only when the exact digit sequence actually appears
    somewhere in the OCR text.
    """
    candidate = evidence.get("cnNumber")
    details = evidence.setdefault("evidence", {})

    if candidate is None:
        return evidence

    normalized_candidate = re.sub(r"\D", "", str(candidate))
    if not normalized_candidate:
        evidence["cnNumber"] = None
        details["cnNumber"] = f"Model returned {candidate!r}, which contains no digits; discarded."
        return evidence

    ocr_numeric_substrings = set()
    for item in ocr_data or []:
        text = str(item.get("text", ""))
        ocr_numeric_substrings.update(re.findall(r"\d{6,20}", text))

    if normalized_candidate not in ocr_numeric_substrings:
        evidence["cnNumber"] = None
        details["cnNumber"] = (
            f"Model returned {candidate!r}; no matching digit run found anywhere "
            "in OCR text, so it was discarded rather than guessed."
        )

    return evidence


def validate_evidence(data):
    required = set(EVIDENCE_SCHEMA["required"])
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("Model returned unexpected top-level evidence fields.")

    bool_fields = (
        "hasSignature", "hasStamp", "hasHandwriting", "imageQualityPassed",
        "physicalDamage", "businessDamage", "shortage", "manualReviewRequired"
    )
    for field in bool_fields:
        if not isinstance(data[field], bool):
            raise ValueError(f"{field} must be a boolean.")

    for field in ("cnNumber", "deliveryDate", "remarksText"):
        if data[field] is not None and not isinstance(data[field], str):
            raise ValueError(f"{field} must be a string or null.")

    score = data["confidenceScore"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise ValueError("confidenceScore must be a number between 0 and 1.")

    evidence = data["evidence"]
    evidence_required = set(EVIDENCE_SCHEMA["properties"]["evidence"]["required"])
    if not isinstance(evidence, dict) or set(evidence) != evidence_required:
        raise ValueError("Model returned unexpected evidence object fields.")
    if not all(isinstance(v, str) for v in evidence.values()):
        raise ValueError("All evidence description fields must be strings.")


_DEGENERATE_EVIDENCE = re.compile(r"^(true|false|yes|no)\.?$", re.I)
_DENIAL_LANGUAGE = re.compile(
    r"\b(unmark|not marked|no mark|not checked|unchecked|blank|empty|"
    r"not visible|no visible|not selected|none visible|not present|"
    r"no stamp visible|no signature visible|nothing (?:filled|written|entered)|"
    r"no explicit|not found|no .{0,20} field found|unclear|ambiguous|"
    r"cannot be (?:confidently|reliably) (?:read|determined|identified))\b",
    re.I,
)


def _evidence_supports_positive(evidence_text):
    """True only if the text is a real, non-contradicting description."""
    text = (evidence_text or "").strip()
    if not text:
        return False
    if _DEGENERATE_EVIDENCE.match(text):
        return False
    if _DENIAL_LANGUAGE.search(text):
        return False
    return True


_LABEL_TOKENS = re.compile(
    r"(short\s*qty\.?|damage|qty|remarks?|recd\s*by\s*s?\.?|signature|short)",
    re.I,
)
_ONLY_SEPARATORS = re.compile(r"^[\s,;.\-:]*$")


def _sanitize_remarks(remarks_text):
    """
    Discard remarksText if, after stripping out every known printed field
    label, nothing but punctuation/whitespace is left -- meaning the model
    just echoed labels back with no actual filled content. Genuine entered
    content (a quantity, a name, a description) survives this stripping
    and is preserved untouched.

    FIX: this was reduced to a no-op (just a strip()), which reopened the
    exact bug fixed earlier -- an echoed label like "Short QTY. Recd By S"
    would pass straight through. That matters even more now, since
    classify_pod's has_positive_term() scans remarksText for the words
    "short"/"damage" directly. An unsanitized echoed label would make
    classify_pod see the word "short" and misclassify the POD as
    ISSUE_POD_SHORT even when no checkbox was actually marked.
    """
    if not isinstance(remarks_text, str):
        return None

    remarks_text = remarks_text.strip()
    if not remarks_text:
        return None

    residual = _LABEL_TOKENS.sub("", remarks_text)
    if _ONLY_SEPARATORS.match(residual):
        return None

    return remarks_text


def classify_pod(evidence):
    """
    Classify PODs using the required priority order.

    Priority:
      1. Image quality failure -> manual review
      2. Physical paper damage -> manual review
      3. Goods damage + shortage
      4. Goods damage
      5. Shortage
      6. Stamp + signature
      7. Stamp only
      8. Signature only
      9. Neither stamp nor signature

    Remarks can provide evidence of business issues, but
    business damage is not the same as physical paper damage.
    """

    import re

    evidence = dict(evidence or {})
    details = evidence.get("evidence") or {}

    if not isinstance(details, dict):
        details = {}

    # ---------------------------------------------------------
    # Helper: safely interpret model evidence
    # ---------------------------------------------------------
    def positive_evidence(field):
        try:
            return _evidence_supports_positive(
                details.get(field)
            )
        except Exception:
            return False

    # ---------------------------------------------------------
    # Helper: normalize text
    # ---------------------------------------------------------
    def normalize_text(value):
        if not isinstance(value, str):
            return ""
        return " ".join(value.lower().split())

    # ---------------------------------------------------------
    # Helper: detect issue terms without obvious negation
    # ---------------------------------------------------------
    def has_positive_term(text, terms):
        """
        Match issue terms while avoiding simple negations such
        as 'no damage', 'not damaged', or 'shortage not found'.
        """
        text = normalize_text(text)

        if not text:
            return False

        for term in terms:
            pattern = (
                r"(?<!\w)"
                + re.escape(term)
                + r"(?!\w)"
            )

            for match in re.finditer(pattern, text):
                start = max(0, match.start() - 40)
                prefix = text[start:match.start()].strip()

                # Ignore a term when its immediate context
                # explicitly denies that issue.
                if re.search(
                    r"\b(?:no|not|without|none|never|"
                    r"absent|否)\s+(?:\w+\s+){0,3}$",
                    prefix
                ):
                    continue

                # Also handle phrases like:
                # "damage not observed"
                # "shortage not detected"
                following = text[
                    match.end():match.end() + 35
                ]

                if re.match(
                    r"\s+(?:was\s+|is\s+|were\s+)?"
                    r"(?:not|never)\s+"
                    r"(?:observed|detected|found|reported|present)\b",
                    following
                ):
                    continue

                return True

        return False

    # ---------------------------------------------------------
    # 1. Image quality
    # ---------------------------------------------------------
    if evidence.get("imageQualityPassed") is not True:
        return (
            "MANUAL_CHECK_REQUIRED",
            "Image quality is insufficient for reliable automated analysis."
        )

    # ---------------------------------------------------------
    # 2. Resolve evidence contradictions
    # ---------------------------------------------------------
    contradictions = []

    boolean_fields = (
        ("hasSignature", "signature"),
        ("hasStamp", "stamp"),
        ("physicalDamage", "physicalDamage"),
        ("businessDamage", "businessDamage"),
        ("shortage", "shortage"),
    )

    for boolean_field, evidence_field in boolean_fields:
        if evidence.get(boolean_field) is True:
            if not positive_evidence(evidence_field):
                contradictions.append(
                    f"{boolean_field} is marked present, "
                    "but its supporting evidence is missing "
                    "or not clearly positive."
                )

                evidence[boolean_field] = False

    # ---------------------------------------------------------
    # 3. Physical paper damage
    # ---------------------------------------------------------
    physical_damage = (
        evidence.get("physicalDamage") is True
    )

    if physical_damage:
        return (
            "MANUAL_CHECK_REQUIRED",
            "The POD paper itself appears physically damaged."
        )

    # ---------------------------------------------------------
    # 4. Determine business issues from structured evidence
    #    and explicit remarks
    # ---------------------------------------------------------
    remarks = normalize_text(
        evidence.get("remarksText")
    )

    shortage_terms = (
        "short",
        "shortage",
        "short qty",
        "short quantity",
    )

    damage_terms = (
        "damage",
        "damaged",
    )

    remarks_has_shortage = has_positive_term(
        remarks,
        shortage_terms
    )

    remarks_has_damage = has_positive_term(
        remarks,
        damage_terms
    )

    shortage = (
        evidence.get("shortage") is True
        or (
            positive_evidence("shortage")
            and not contradictions
        )
        or remarks_has_shortage
    )

    business_damage = (
        evidence.get("businessDamage") is True
        or (
            positive_evidence("businessDamage")
            and not contradictions
        )
        or remarks_has_damage
    )

    # ---------------------------------------------------------
    # 5. Manual review flag
    # ---------------------------------------------------------
    if evidence.get("manualReviewRequired") is True:
        model_review = normalize_text(
            details.get("manualReview")
        )

        reasons = list(contradictions)

        if model_review:
            reasons.append(model_review)

        if not reasons:
            reasons.append(
                "The model flagged an uncertain POD field or visual mark."
            )

        return (
            "MANUAL_CHECK_REQUIRED",
            "Manual review required: " + " ".join(reasons)
        )

    # ---------------------------------------------------------
    # 6. Business issue classification
    # ---------------------------------------------------------
    if business_damage and shortage:
        return (
            "ISSUE_POD_DAMAGED_AND_SHORT",
            "Evidence indicates both goods damage and shortage."
        )

    if business_damage:
        return (
            "ISSUE_POD_DAMAGED",
            "Evidence indicates goods or material damage."
        )

    if shortage:
        return (
            "ISSUE_POD_SHORT",
            "Evidence indicates a shortage."
        )

    # ---------------------------------------------------------
    # 7. Clean POD classification
    # ---------------------------------------------------------
    has_stamp = (
        evidence.get("hasStamp") is True
    )

    has_signature = (
        evidence.get("hasSignature") is True
    )

    if has_stamp and has_signature:
        return (
            "CLEAN_POD_SEAL_AND_SIGNATURE",
            "A recognizable stamp and signature are present."
        )

    if has_stamp:
        return (
            "CLEAN_POD_ONLY_SEAL",
            "A recognizable stamp is present; no signature was identified."
        )

    if has_signature:
        return (
            "CLEAN_POD_ONLY_SIGNATURE",
            "A recognizable signature is present; no stamp was identified."
        )

    return (
        "NO_SIGNATURE_NO_STAMP",
        "No recognizable signature or stamp was identified."
    )


def _ocr_supports_date(delivery_date_iso, ocr_data):
    """
    A YYYY-MM-DD delivery date is only trusted if the same day+month
    genuinely appears somewhere in the OCR text, AND that OCR fragment's
    year has as many digits as the year we're trusting.
    """
    try:
        year, month, day = (int(p) for p in delivery_date_iso.split("-"))
    except (ValueError, AttributeError, TypeError):
        return False

    for item in ocr_data or []:
        text = str(item.get("text", ""))
        for d, mon_str, y in _OCR_TEXT_DATE.findall(text):
            mon = _MONTH_ABBR.get(mon_str.upper()[:3])
            if mon == month and int(d) == day and len(y) == 4:
                return True
        for a, b, y in _OCR_NUMERIC_DATE.findall(text):
            if len(y) == 4 and int(a) == day and int(b) == month:
                return True
    return False


def _extract_labeled_cn_from_ocr(ocr_data):
    """
    Return a CN only when OCR contains an explicit Docket/CN label followed
    immediately by one unambiguous numeric token. Never use unrelated numbers.
    """
    if not ocr_data:
        return None
    candidates = set()
    for item in ocr_data:
        text = str(item.get("text", "") or "")
        match = re.search(
            r"\b(?:docket|cn\s*(?:no\.?|number)?|consignment\s*(?:no\.?|number)?)"
            r"\s*[:#-]?\s*(\d{8,18})\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            candidates.add(match.group(1))
    return next(iter(candidates)) if len(candidates) == 1 else None


def build_final_output(evidence, ocr_data=None, preprocess_meta=None):
    evidence = dict(evidence)
    evidence_details = evidence.get("evidence") or {}
    evidence["evidence"] = evidence_details

    # FIX: sanitize remarksText BEFORE classification, not after.
    # classify_pod() derives shortage/businessDamage directly from
    # remarksText (has_positive_term), so an unsanitized echoed label
    # ("Short QTY. Recd By S") would previously slip straight into the
    # classifier and trigger a false ISSUE_POD_SHORT/DAMAGED result. Now
    # the classifier only ever sees remarks that survived sanitization.
    evidence["remarksText"] = _sanitize_remarks(evidence.get("remarksText"))

    cn_number = evidence.get("cnNumber")
    if not cn_number:
        cn_number = _extract_labeled_cn_from_ocr(ocr_data)

    delivery_date = evidence.get("deliveryDate")

    if delivery_date and not _evidence_supports_positive(
        evidence_details.get("deliveryDate")
    ):
        delivery_date = None

    if delivery_date and not _ocr_supports_date(delivery_date, ocr_data):
        delivery_date = None

    if delivery_date:
        try:
            datetime.strptime(delivery_date, "%Y-%m-%d")
        except (TypeError, ValueError):
            delivery_date = None

    category, reason = classify_pod(evidence)

    require_date_review = os.getenv(
        "POD_REQUIRE_DELIVERY_DATE_FOR_AUTO_CLASSIFICATION", "false"
    ).strip().lower() in {"1", "true", "yes"}
    review_reasons = []
    preprocess_meta = preprocess_meta or {}
    if preprocess_meta.get("orientation_uncertain"):
        review_reasons.append("Automatic page orientation was ambiguous.")
    if preprocess_meta.get("boundary_detected") is False:
        review_reasons.append("Document boundary was not confidently detected; full frame was retained.")

    review_text = str(evidence_details.get("manualReview", "") or "").lower()
    uncertainty_terms = (
        "overlap", "overlapping", "occluded", "obscured", "covered",
        "multiple documents", "ambiguous", "partially visible", "cut off",
    )
    if any(term in review_text for term in uncertainty_terms):
        review_reasons.append(
            "The model reported overlapping or obscured document content."
        )

    if require_date_review and not delivery_date:
        review_reasons.append(
            "Delivery date is required by configuration but could not be verified."
        )

    if not cn_number:
        pass

    if review_reasons:
        category = "MANUAL_CHECK_REQUIRED"
        reason = "Manual review required: " + " ".join(review_reasons)

    remarks_text = evidence["remarksText"]  # already sanitized above

    confidence = float(evidence.get("confidenceScore", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    if category == "MANUAL_CHECK_REQUIRED":
        confidence = min(confidence, 0.70)

    return {
        "cnNumber": cn_number,
        "hasSignature": bool(evidence.get("hasSignature", False)),
        "hasStamp": bool(evidence.get("hasStamp", False)),
        "hasHandwriting": bool(evidence.get("hasHandwriting", False)),
        "imageQualityPassed": bool(evidence.get("imageQualityPassed", False)),
        "remarksText": remarks_text,
        "deliveryDate": delivery_date,
        "categoryReason": reason,
        "confidenceScore": round(confidence, 3),
        "podCategory": category,
        "limit_exceed": False,
    }


def display_json(title, data):
    print(f"\n========== {title} ==========")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("=" * (len(title) + 22))


if __name__ == "__main__":
    if not os.path.isfile(IMAGE_PATH):
        raise SystemExit(
            f"ERROR: POD image not found: {IMAGE_PATH}\n"
            "Set POD_IMAGE_PATH in .env or update the path."
        )

    print(f"Input image: {IMAGE_PATH}")
    print(f"Rotation setting: {POD_ROTATION_SETTING!r}")
    processed = preprocess_image(IMAGE_PATH)

    ocr_result = run_ocr(processed["enhanced"])
    ocr_data = extract_ocr_data(ocr_result)

    # FIX: in "auto" mode, orientation is now detected here, from the same
    # OCR call, instead of via 4 separate probe calls beforehand. If a
    # confident rotation was found, apply it to the saved color "document"
    # image too, so the vision model (Qwen) sees the same upright page that
    # OCR actually read. A manual override was already applied earlier in
    # preprocess_image, so there is nothing more to do here in that case.
    orientation_uncertain = False
    if POD_ROTATION_SETTING == "auto":
        detected_angle = extract_orientation_angle(ocr_result)
        orientation_uncertain = detected_angle is None
        if detected_angle:
            print(f"Detected page rotation: {detected_angle} degrees. "
                  f"Rotating the saved document and enhanced images to match.")
            # FIX: previously only the color "document" image was rotated
            # and re-saved here. The "enhanced" image (what OCR actually
            # reads) never got the same treatment, so it stayed stale on
            # disk -- confusing for debugging, even though PaddleOCR's
            # internal correction meant the OCR text itself was still
            # read correctly. Now both saved files reflect the same
            # final orientation.
            for key in ("document", "enhanced"):
                img = cv2.imread(processed[key], cv2.IMREAD_UNCHANGED)
                if img is not None:
                    img = _rotate_candidate(img, detected_angle)
                    cv2.imwrite(processed[key], img)
        elif orientation_uncertain:
            print("WARNING: Orientation classifier did not return a confident angle; "
                  "sending the image as-is.")

    normalized_ocr = normalize_ocr_data(ocr_data)
    display_normalized_ocr(normalized_ocr)
    ocr_text = build_ocr_text(normalized_ocr)

    print("\n========== OCR TEXT ==========")
    print(ocr_text)
    print("==============================")

    evidence = run_vision_model(processed["document"], ocr_text)
    evidence = reconcile_cn_with_ocr(evidence, ocr_data)
    display_json("QWEN STRUCTURED EVIDENCE", evidence)

    preprocess_meta = {
        "boundary_detected": processed.get("boundary_detected"),
        "orientation_uncertain": orientation_uncertain,
    }
    final_output = build_final_output(evidence, ocr_data, preprocess_meta)
    display_json("FINAL POD CLASSIFICATION JSON", final_output)