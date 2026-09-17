# ============================================================
# POD OCR TEST
# PaddleOCR + Image Preprocessing + Local Ollama Information Model
# Windows CPU (OCR) / Local Ollama (information extraction)
# ============================================================

import os
import json
import re
import requests

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

# Explicitly disable MKLDNN
paddle.set_flags({
    "FLAGS_use_mkldnn": False
})


# ============================================================
# MODEL CONFIG
# ============================================================

# Read the model tag from .env (INFORMATION_MODEL=gemma3n:e4b), falling
# back to a sane default. Run `ollama list` to confirm the exact tag you
# actually pulled — a silently wrong tag can behave oddly instead of
# failing cleanly.
INFORMATION_MODEL = os.getenv("INFORMATION_MODEL") or "gemma3n:e4b"


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    print("\n========== PREPROCESSING ==========\n")

    image = Image.open(image_path)

    print(f"Original image size: {image.size}")

    # Fix EXIF orientation
    image = ImageOps.exif_transpose(image)

    # Convert to RGB
    image = image.convert("RGB")

    # Improve contrast
    image = ImageEnhance.Contrast(image).enhance(1.2)

    # Improve sharpness
    image = ImageEnhance.Sharpness(image).enhance(1.3)

    # Save processed image
    os.makedirs("processed_images", exist_ok=True)

    filename = os.path.basename(image_path)

    output_path = os.path.join("processed_images", filename)

    image.save(output_path, quality=95)

    print(f"Processed image saved to: {output_path}")

    return output_path


# ============================================================
# OCR MODEL
# ============================================================

def run_ocr(image_path):

    print("\n========== LOADING OCR MODEL ==========\n")

    ocr = PaddleOCR(lang="en", device="cpu")

    print("\n========== RUNNING OCR ==========\n")

    result = ocr.predict(image_path)

    return result


# ============================================================
# EXTRACT OCR WITH POSITION INFORMATION
# ============================================================

def extract_ocr_data(result):
    """
    Extract OCR text together with confidence and bounding box.
    """

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

            if not text:
                continue

            # Ignore very low confidence OCR
            if score < 0.50:
                continue

            all_data.append({
                "text": text,
                "confidence": float(score),
                "box": box.tolist() if hasattr(box, "tolist") else box,
            })

    return all_data


# ============================================================
# NORMALIZE OCR BOUNDING BOXES
# ============================================================

def normalize_ocr_data(ocr_data):
    """
    Convert PaddleOCR polygon boxes into simple x/y coordinates.
    """

    normalized = []

    for item in ocr_data:

        box = item["box"]

        xs = [point[0] for point in box]
        ys = [point[1] for point in box]

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

        width = x2 - x1
        height = y2 - y1

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        normalized.append({
            "text": item["text"],
            "confidence": round(item["confidence"], 4),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": round(center_x, 2),
            "center_y": round(center_y, 2),
            "width": width,
            "height": height,
        })

    return normalized


# ============================================================
# DISPLAY NORMALIZED OCR
# ============================================================

def display_normalized_ocr(ocr_data):

    print("\n========== NORMALIZED OCR DATA ==========\n")

    print(
        f"{'TEXT':<40} "
        f"{'SCORE':<8} "
        f"{'CENTER':<20} "
        f"{'SIZE':<15}"
    )

    print("-" * 100)

    for item in ocr_data:

        print(
            f"{item['text'][:38]:<40} "
            f"{item['confidence']:<8.3f} "
            f"({item['center_x']:.0f}, "
            f"{item['center_y']:.0f})"
            f"{'':<7}"
            f"{item['width']:.0f} x "
            f"{item['height']:.0f}"
        )

    print("\n==========================================\n")


# ============================================================
# BUILD OCR TEXT
# ============================================================

def build_ocr_text(ocr_data):
    """Convert OCR detections into clean text for the information model."""

    lines = []

    for item in ocr_data:
        text = item["text"].strip()

        if not text:
            continue

        lines.append(text)

    return "\n".join(lines)


# ============================================================
# DETERMINISTIC OCR CANDIDATES
# ============================================================

def _candidate_from_line(value, line, line_id, label=None, confidence=0.65,
                          reason="Deterministic OCR text candidate; association requires review."):
    return {
        "value": value,
        "associated_label": label,
        "evidence_text": line,
        "line_ids": [line_id],
        "confidence": confidence,
        "reason": reason,
    }


def add_deterministic_ocr_candidates(data, ocr_lines):
    """Supplement model output with text-grounded candidates without asserting field identity."""

    def add_unique(field, candidate):
        key = candidate["value"].strip().casefold()
        if not any(
            str(item.get("value", "")).strip().casefold() == key
            for item in data.get(field, [])
            if isinstance(item, dict)
        ):
            data.setdefault(field, []).append(candidate)

    for item in ocr_lines:
        text = item["text"]
        line_id = item["line_id"]
        upper = text.upper()

        # Manual references are retained separately and never promoted to CN here.
        if re.search(r"MANUAL\s*(?:REF(?:ERENCE)?|NO\.?|NUMBER)", upper):
            numbers = re.findall(r"(?<!\d)\d{8,}(?!\d)", text)
            for number in numbers:
                add_unique("manual_reference_candidates", _candidate_from_line(
                    number, text, line_id, "MANUAL REF", 0.95,
                    "Long numeric value occurs on a line explicitly labeled as a manual reference."))

        # Capture long numeric strings as uncertain identifier candidates unless the
        # same line clearly labels them as invoice/e-waybill/manual reference.
        excluded = re.search(r"MANUAL\s*REF|INVOICE|E.?WAY.?BILL|GST|PHONE|MOBILE|PIN", upper)
        if not excluded:
            for number in re.findall(r"(?<!\d)\d{10,}(?!\d)", text):
                add_unique("cn_candidates", _candidate_from_line(
                    number, text, line_id, None, 0.55,
                    "Unlabeled long numeric OCR candidate only; not confirmed as CN."))

        # Preserve all date-like strings as candidates; do not repair OCR corruption.
        for match in re.finditer(r"(?<!\d)(?:\d{1,2}[./-]){1,2}\d{2,4}(?:\s*\d{1,2}:?\d{0,2})?(?!\d)", text):
            date_value = match.group(0).strip()
            add_unique("date_candidates", {
                **_candidate_from_line(date_value, text, line_id, None, 0.6,
                    "Date-like OCR text preserved verbatim; date meaning is unconfirmed."),
                "type": "unknown",
            })

        # Preserve issue words as text evidence, not as damage/short classification.
        for word in re.findall(r"\b(?:SHORT|DAMAGE|DAMAGED|SHORTAGE)\b(?:\s+QTY\.? )?", text, flags=re.I):
            value = word.strip()
            add_unique("remark_candidates", {
                **_candidate_from_line(value, text, line_id, None, 0.8,
                    "Issue-related word detected in OCR; context and meaning are not inferred."),
                "type": "issue_word_candidate",
            })

    return data


# ============================================================
# INFORMATION EXTRACTION MODEL (LOCAL OLLAMA)
# ============================================================

def run_information_model(ocr_text):
    """Extract OCR-grounded candidates using local Ollama and validate them."""

    ocr_lines = [
        {"line_id": i + 1, "text": line.strip()}
        for i, line in enumerate(ocr_text.splitlines())
        if line.strip()
    ]

    expected = {
        "cn_candidates",
        "date_candidates",
        "consignor_candidates",
        "consignee_candidates",
        "invoice_candidates",
        "ewaybill_candidates",
        "remark_candidates",
        "manual_reference_candidates",
    }

    valid_date_types = {
        "pickup_date",
        "delivery_date_candidate",
        "shipping_date",
        "invoice_date",
        "mr_date",
        "unknown",
    }

    valid_remark_types = {
        "remark_candidate",
        "issue_word_candidate",
    }

    # ---------- JSON SCHEMA ----------

    candidate_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "associated_label": {"type": ["string", "null"]},
            "evidence_text": {"type": "string"},
            "line_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "reason": {"type": "string"},
        },
        "required": [
            "value", "associated_label", "evidence_text",
            "line_ids", "confidence", "reason",
        ],
        "additionalProperties": False,
    }

    date_candidate_schema = {
        **candidate_schema,
        "properties": {
            **candidate_schema["properties"],
            "type": {"type": "string", "enum": sorted(valid_date_types)},
        },
        "required": candidate_schema["required"] + ["type"],
    }

    remark_candidate_schema = {
        **candidate_schema,
        "properties": {
            **candidate_schema["properties"],
            "type": {"type": "string", "enum": sorted(valid_remark_types)},
        },
        "required": candidate_schema["required"] + ["type"],
    }

    schema = {
        "type": "object",
        "properties": {
            "cn_candidates": {"type": "array", "items": candidate_schema},
            "date_candidates": {"type": "array", "items": date_candidate_schema},
            "consignor_candidates": {"type": "array", "items": candidate_schema},
            "consignee_candidates": {"type": "array", "items": candidate_schema},
            "invoice_candidates": {"type": "array", "items": candidate_schema},
            "ewaybill_candidates": {"type": "array", "items": candidate_schema},
            "remark_candidates": {"type": "array", "items": remark_candidate_schema},
            "manual_reference_candidates": {"type": "array", "items": candidate_schema},
        },
        "required": sorted(expected),
        "additionalProperties": False,
    }

    # ---------- PROMPT ----------

    prompt = f"""
You are the OCR interpretation stage of a Proof of Delivery (POD)
pipeline. You receive OCR text only, not the image.

Extract candidate information. Do not classify the POD, visually
verify anything, or guess.

GENERAL
- OCR may be noisy, reordered, incomplete, merged, or duplicated.
- Preserve multiple plausible candidates.
- Every candidate must cite exact OCR evidence and supplied line IDs.
- Preserve values exactly as OCR. Do not repair digits or dates.
- Confidence measures textual support, not visual truth.
- Never invent values. Use [] when no candidate is supported.

CN / DOCKET
- Look for CN, CN No, CNR, consignment, docket, or waybill labels.
- Include a value as a CN candidate only when the text supports
  that association.
- Manual Ref / Manual Reference belongs ONLY in
  manual_reference_candidates unless separately labeled as CN.
- Exclude GST, phone, PIN, transporter ID, invoice/e-waybill,
  amounts, weights, party/store numbers, and unrelated references.

DATES
- Extract all supported dates.
- Use pickup_date, delivery_date_candidate, shipping_date,
  invoice_date, mr_date, or unknown only where supported.
- Preserve malformed or truncated dates exactly.
- Never complete missing digits.

PARTIES
- Associate consignor/consignee only when supported by text.
- Do not assume Billing Party, Bill To, transporter, source,
  or destination is consignor/consignee without explicit evidence.
- Do not duplicate a company into both fields without evidence.

INVOICE / E-WAYBILL
- Require a relevant label.
- Preserve ambiguity with lower confidence and explain it.
- Do not mistake amounts or CNs for these identifiers.

REMARKS
- Extract delivery notes/text under REMARKS where supported.
- Exclude generic legal boilerplate when distinguishable.
- Preserve Short/Damage as text only.
- Do not infer selected options, goods damage, shortage,
  or physical paper damage.

Return exactly the eight keys required by the JSON schema.
Every candidate must include all required fields.
Date candidates must include a valid date type.
Remark candidates must include a valid remark type.
Do not add extra keys.

OCR lines with provenance:
{json.dumps(ocr_lines, ensure_ascii=False, indent=2)}
"""

    print("\n========== INFORMATION MODEL ==========")
    print(f"Model: local Ollama {INFORMATION_MODEL}")

    # ---------- CALL OLLAMA ----------

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": INFORMATION_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You extract POD candidate fields from OCR text. "
                            "Return only the JSON object required by the "
                            "provided JSON schema. Do not classify or guess. "
                            "Use only supplied OCR lines and line IDs."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "stream": False,
                "format": schema,
                # FIX: disable the reasoning trace so the full token budget
                # goes toward the JSON answer instead of being burned on
                # "thinking" tokens that were causing truncation.
                "think": False,
                "options": {
                    "temperature": 0,
                    # FIX: default context (4096) was smaller than the
                    # prompt itself (~4018 tokens), leaving almost no room
                    # for a response. Raised to give real headroom.
                    "num_ctx": 8192,
                    # FIX: explicit cap so the response has guaranteed
                    # room to complete instead of relying on defaults.
                    "num_predict": 2048,
                },
            },
            timeout=180,
        )

        response.raise_for_status()

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "Could not reach local Ollama. "
            "Make sure Ollama is running and the model is available."
        ) from exc

    # ---------- INSPECT OLLAMA RESPONSE ----------

    response_data = response.json()

    print("\n========== OLLAMA RESPONSE DEBUG ==========")
    print(json.dumps(response_data, indent=2, ensure_ascii=False))
    print("===========================================")

    message = response_data.get("message") or {}
    content = message.get("content") or ""

    if not content.strip():
        raise RuntimeError(
            "Ollama returned empty message.content. "
            f"done_reason={response_data.get('done_reason')!r}, "
            f"message_keys={list(message.keys())}. "
            "If this still happens after the num_ctx/think fix, your "
            "GPU likely can't hold the model at this context size — "
            "try shrinking the prompt instead of raising num_ctx further."
        )

    # ---------- PARSE JSON ----------

    try:
        data = json.loads(content)

    except json.JSONDecodeError as exc:
        print("\n========== RAW INFORMATION MODEL OUTPUT ==========")
        print(content)
        print("===================================================")

        raise RuntimeError(
            "Information Model did not return valid JSON."
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("Information Model output must be a JSON object.")

    if set(data.keys()) != expected:
        print("\n========== RAW MODEL RESPONSE ==========")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("========================================")

        raise RuntimeError(
            "Model returned an unexpected JSON schema. "
            f"Expected: {sorted(expected)}; "
            f"Received: {sorted(data.keys())}"
        )

    # ---------- ADD DETERMINISTIC OCR CANDIDATES ----------

    data = add_deterministic_ocr_candidates(data, ocr_lines)

    valid_ids = {line["line_id"] for line in ocr_lines}

    required_fields = {
        "value", "associated_label", "evidence_text",
        "line_ids", "confidence", "reason",
    }

    # ---------- VALIDATE CANDIDATES ----------

    for field, candidates in data.items():

        if not isinstance(candidates, list):
            raise RuntimeError(f"{field} must be a list.")

        for i, candidate in enumerate(candidates):

            if not isinstance(candidate, dict):
                raise RuntimeError(f"{field}[{i}] must be an object.")

            missing = required_fields - set(candidate.keys())

            if missing:
                raise RuntimeError(f"{field}[{i}] is missing fields: {sorted(missing)}")

            if not isinstance(candidate["value"], str) or not candidate["value"].strip():
                raise RuntimeError(f"{field}[{i}].value must be non-empty text.")

            label = candidate["associated_label"]

            if label is not None and not isinstance(label, str):
                raise RuntimeError(f"{field}[{i}].associated_label must be a string or null.")

            if not isinstance(candidate["reason"], str):
                raise RuntimeError(f"{field}[{i}].reason must be a string.")

            ids = candidate["line_ids"]

            if (
                not isinstance(ids, list)
                or not ids
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in ids)
                or not all(x in valid_ids for x in ids)
            ):
                raise RuntimeError(f"{field}[{i}] contains invalid or empty line_ids.")

            cited_text = "\n".join(
                line["text"] for line in ocr_lines if line["line_id"] in ids
            )

            evidence = candidate["evidence_text"]

            if (
                not isinstance(evidence, str)
                or not evidence.strip()
                or evidence.strip() not in cited_text
            ):
                raise RuntimeError(
                    f"{field}[{i}].evidence_text does not match its cited OCR line_ids."
                )

            score = candidate["confidence"]

            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0 <= score <= 1
            ):
                raise RuntimeError(f"{field}[{i}].confidence must be between 0 and 1.")

            # Defensive validation for date candidates.
            if field == "date_candidates":
                if candidate.get("type") not in valid_date_types:
                    candidate["type"] = "unknown"

            # Defensive validation for remark candidates.
            if field == "remark_candidates":
                if candidate.get("type") not in valid_remark_types:
                    candidate["type"] = "remark_candidate"

    print("\nInformation Model extraction and validation completed.")

    return data


def display_information_model_output(data):
    """Pretty-print the structured information model result."""

    print("\n========== INFORMATION MODEL OUTPUT ==========")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("===============================================\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_path = "images/pod4.jpg"

    if not os.path.exists(image_path):
        print(f"\nERROR: Image not found: {image_path}")
        raise SystemExit(1)

    # --------------------------------------------------------
    # STEP 1: PREPROCESSING
    # --------------------------------------------------------

    processed_image = preprocess_image(image_path)

    # --------------------------------------------------------
    # STEP 2: OCR
    # --------------------------------------------------------

    ocr_result = run_ocr(processed_image)

    # --------------------------------------------------------
    # STEP 3: EXTRACT OCR DATA
    # --------------------------------------------------------

    ocr_data = extract_ocr_data(ocr_result)

    # --------------------------------------------------------
    # STEP 4: NORMALIZE COORDINATES
    # --------------------------------------------------------

    normalized_ocr = normalize_ocr_data(ocr_data)

    # --------------------------------------------------------
    # STEP 5: DISPLAY OCR
    # --------------------------------------------------------

    display_normalized_ocr(normalized_ocr)

    # --------------------------------------------------------
    # STEP 6: BUILD RAW OCR TEXT
    # --------------------------------------------------------

    ocr_text = build_ocr_text(ocr_data)

    print("\n========== RAW OCR TEXT ==========")
    print(ocr_text)
    print("==================================\n")

    # --------------------------------------------------------
    # STEP 7: INFORMATION EXTRACTION MODEL
    # --------------------------------------------------------

    print(f"========== INFORMATION MODEL ==========\nModel: {INFORMATION_MODEL}\n")

    information = run_information_model(ocr_text)

    # --------------------------------------------------------
    # STEP 8: DISPLAY STRUCTURED INFORMATION
    # --------------------------------------------------------

    display_information_model_output(information)