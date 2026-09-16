import os
import base64

# --------------------------------------------------
# PADDLE / CPU SETTINGS
# --------------------------------------------------

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"

import paddle

paddle.set_flags({
    "FLAGS_use_mkldnn": False
})

from paddleocr import PaddleOCR
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageOps


# --------------------------------------------------
# LOAD ENVIRONMENT
# --------------------------------------------------

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image_path):

    image = Image.open(image_path)

    # Automatically correct EXIF orientation
    image = ImageOps.exif_transpose(image)

    # Convert to RGB
    image = image.convert("RGB")

    # Slight contrast improvement
    image = ImageEnhance.Contrast(image).enhance(1.2)

    # Slight sharpness improvement
    image = ImageEnhance.Sharpness(image).enhance(1.3)

    # Create folder
    os.makedirs("processed_images", exist_ok=True)

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        "processed_images",
        filename
    )

    image.save(output_path, quality=95)

    print(f"Processed image saved to: {output_path}")

    return output_path


# --------------------------------------------------
# OCR
# --------------------------------------------------

def run_ocr(image_path):

    print("\n========== RUNNING OCR ==========\n")

    ocr = PaddleOCR(
        lang="en",
        device="cpu"
    )

    result = ocr.predict(image_path)

    extracted_text = []

    for page in result:

        # PaddleOCR result object
        data = page.json

        if callable(data):
            data = data()

        # Get result dictionary
        res = data.get("res", data)

        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])

        for text, score in zip(texts, scores):

            if not text:
                continue

            # Ignore very low-confidence OCR
            if score < 0.50:
                continue

            text = text.strip()

            if text:
                extracted_text.append(text)

    ocr_text = "\n".join(extracted_text)

    print("========== EXTRACTED OCR TEXT ==========\n")
    print(ocr_text)
    print("\n=========================================\n")

    return ocr_text


# --------------------------------------------------
# IMAGE → BASE64 DATA URL
# --------------------------------------------------

def image_to_data_url(image_path):

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    encoded = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# --------------------------------------------------
# QWEN PROMPT
# --------------------------------------------------

def build_prompt(ocr_text):

    return f"""
You are an expert Proof of Delivery (POD) document analysis and
information extraction system.

Your task is to carefully analyze the provided POD image and return
structured evidence as JSON.

You have TWO sources of information:

SOURCE 1:
The actual POD image.

SOURCE 2:
OCR text extracted from the POD.

The POD IMAGE is the primary source of truth.
OCR is only supporting evidence.

OCR may contain:
- incorrect characters
- missing words
- incorrect numbers
- merged fields
- duplicated text
- incorrect orientation
- false detections

Therefore, NEVER blindly trust OCR.
Always verify important information against the actual image.

============================================================
OCR SUPPORTING EVIDENCE
============================================================

{ocr_text}

============================================================
GENERAL RULES
============================================================

1. Analyze the ENTIRE POD image.
2. Inspect the document even if it is rotated.
3. Do not guess.
4. Do not hallucinate.
5. Do not infer a value simply because it looks plausible.
6. If information cannot be reliably identified, return null.
7. Printed labels are NOT evidence by themselves.
8. A field value must be associated with its corresponding label.
9. When OCR and the image disagree, trust the image.
10. Carefully distinguish printed text from handwriting, stamps,
    signatures, checkmarks, and actual remarks.

============================================================
STEP 1 — CN / CONSIGNMENT NUMBER
============================================================

Find the actual CN / consignment / docket number.

Search the entire document for labels such as:

- CN
- CN No
- CN Number
- CNR
- Consignment No
- Consignment Number
- Docket
- Docket No
- Docket Number

A number directly associated with one of these labels is strong
evidence for the CN.

IMPORTANT CN RULES:

The CN must be associated with an appropriate CN/consignment/docket
label.

Do NOT select a number merely because it:
- is large
- appears near the top
- appears near a barcode
- looks like an identifier
- is the only 5 or 6 digit number
- appears near Consignee or Consignor

Do NOT confuse the CN with:

- Consignee name
- Consignor name
- Consignee reference
- Consignor reference
- PIN code
- GST number
- Phone number
- Transporter ID
- Invoice number
- E-Waybill number
- Manual Reference Number
- Party number
- Store number
- Amount
- Weight
- Barcode number
- Date

For example:

"CNR : 787786 KHODIYAR CASTECH"

is evidence that the value associated with CNR is the relevant
document identifier.

A separate value such as:

"360022"

must NOT be selected as CN unless the document explicitly
associates 360022 with CN/CNR/Consignment/Docket.

If no reliable CN can be identified:

"cnNumber": null

NEVER invent or reconstruct missing digits.

============================================================
STEP 2 — SIGNATURE
============================================================

Inspect all signature areas and the entire POD.

Set:

hasSignature = true

ONLY when an actual recognizable handwritten signature is visible.

Do NOT count these as a signature by themselves:

- printed person's name
- printed word "Signature"
- printed "Consignee Sign."
- printed "Sign & Seal"
- blank signature field
- typed text

A handwritten signature counts as handwriting as well.

============================================================
STEP 3 — STAMP / SEAL
============================================================

Inspect the entire document for actual stamps or seals.

Set:

hasStamp = true

ONLY when a visible official/company stamp, seal, or stamped mark
is actually present.

Do NOT mark true simply because the document contains:

"Sign & Seal"
"Consignee Sign. & Seal"
"Stamp"
or another printed label.

The physical stamp/seal itself must be visible.

============================================================
STEP 4 — HANDWRITING
============================================================

Set:

hasHandwriting = true

when ANY actual handwritten information is visible anywhere on
the POD.

Examples include:

- handwritten signature
- handwritten date
- handwritten quantity
- handwritten number
- handwritten note
- handwritten remark
- handwritten correction
- handwritten initials

Printed text does NOT count as handwriting.

A handwritten signature also counts as handwriting.

============================================================
STEP 5 — REMARKS
============================================================

Identify the ACTUAL remarks associated with delivery/receipt of
the shipment.

Look for sections such as:

- Remarks
- Delivery Remarks
- Receiving Remarks
- Received Remarks
- Delivery Comments
- Comments
- Short Qty
- Damage
- Shortage
- Received Short
- Received Damage

IMPORTANT:

Do NOT automatically include general Terms & Conditions text
as remarks.

For example, statements such as:

"At Owner Risk."
"Subject to Delhi Jurisdiction only."

may be printed Terms & Conditions rather than delivery remarks.

Only include them in remarksText if they are clearly part of
the actual delivery/receipt remarks.

IMPORTANT CHECKBOX RULE:

Words such as:

"Short"
"Damage"
"Shortage"

may simply be PRINTED LABELS next to checkboxes.

The presence of the word alone does NOT mean the issue occurred.

You must inspect the corresponding checkbox/mark.

For checkboxes:

- clearly checked/ticked/marked → selected
- clearly empty/unchecked → not selected
- unclear → do not assume selected

Handwritten marks near a Short or Damage option must also be
inspected visually.

If actual relevant remarks are present, return the readable
remark text.

If there are no reliable delivery remarks:

"remarksText": null

Do not hallucinate handwritten text.

============================================================
STEP 6 — BUSINESS DAMAGE
============================================================

Determine whether the GOODS / MATERIAL / SHIPMENT were damaged.

Set:

businessDamage = true

ONLY when the POD clearly indicates that the goods/material were
damaged.

Valid evidence may include:

- an explicitly selected Damage checkbox
- handwritten damage indication
- explicit delivery remark stating damage
- explicit statement that goods/material were damaged

IMPORTANT:

The printed word "Damage" by itself is NOT evidence.

An empty Damage checkbox is NOT evidence.

General T&C text containing the word "damage" is NOT automatically
evidence of shipment damage.

============================================================
STEP 7 — SHORTAGE
============================================================

Determine whether GOODS / MATERIAL / QUANTITY were received short.

Set:

shortage = true

ONLY when the POD clearly indicates a shortage.

Valid evidence may include:

- selected Short checkbox
- Short Qty value
- handwritten shortage indication
- explicit delivery remark stating shortage
- explicit statement that quantity was received short

IMPORTANT:

The printed word "Short" by itself is NOT evidence.

An empty Short checkbox is NOT evidence.

Do not infer shortage simply because quantity information exists.

============================================================
STEP 8 — PHYSICAL POD DAMAGE
============================================================

Determine whether the PHYSICAL PAPER DOCUMENT itself is damaged.

Set:

physicalDamage = true

ONLY if the actual POD paper/document is:

- torn
- ripped
- missing a section
- severely damaged
- destroyed

Do NOT classify these as physical damage:

- normal folds
- wrinkles
- shadows
- slight bending
- perspective distortion
- rotation
- handwritten marks
- stamps
- ink marks

IMPORTANT:

Physical damage to the POD paper is different from damage to
the goods/material.

============================================================
STEP 9 — DELIVERY DATE
============================================================

Find the actual DELIVERY DATE.

Look for fields explicitly associated with:

- Delivery Date
- Delivered Date
- Delivery
- EOD
- POD Date
- Date of Delivery

The date must be associated with a delivery-related field.

DO NOT select a date merely because:

- it is the latest date
- it appears near the bottom
- it appears near a signature
- it is handwritten
- it looks like a delivery date

DO NOT confuse delivery date with:

- Pick Up Date
- Pickup Date
- Shipping Date
- Booking Date
- Invoice Date
- Invoice/Inv Date
- MR Date
- other unrelated dates

If an explicitly labeled delivery date is present, normalize it to:

YYYY-MM-DD

Example:

20-JUL-2026

becomes:

2026-07-20

If the delivery date cannot be reliably identified:

"deliveryDate": null

============================================================
STEP 10 — IMAGE QUALITY
============================================================

Determine whether the POD image is sufficiently clear for reliable
analysis.

Set:

imageQualityPassed = false

if the document is so:

- blurry
- dark
- overexposed
- cropped
- low resolution
- obscured

that important information cannot be reliably analyzed.

Do NOT fail the image merely because it is:

- rotated
- folded
- slightly wrinkled
- photographed at an angle
- affected by normal shadows

============================================================
STEP 11 — FINAL CROSS-CHECK
============================================================

Before generating the JSON, perform a final verification.

CN:
- Is the CN actually associated with CN/CNR/Docket/Consignment?
- Did you accidentally select a PIN, invoice, GST, phone number,
  manual reference, or other identifier?

Signature:
- Is there an actual handwritten signature?

Stamp:
- Is there an actual visible stamp/seal?

Handwriting:
- Is actual handwriting visible?

Remarks:
- Are these actual delivery/receipt remarks?
- Did you accidentally include Terms & Conditions?

Damage:
- Is the Damage option actually selected?
- Is there actual evidence that goods/material were damaged?
- Did you mistake a printed "Damage" label for an issue?

Shortage:
- Is the Short option actually selected?
- Is there actual evidence of shortage?
- Did you mistake a printed "Short" label for an issue?

Physical damage:
- Is the PAPER itself physically damaged?

Delivery date:
- Is the date explicitly associated with delivery?
- Did you accidentally select Pickup/Shipping/Invoice/other date?

OCR:
- Did you verify important OCR values against the image?

Never guess.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Do not provide explanations.
Do not provide Markdown.
Do not use code fences.
Do not add additional fields.
Do not add comments.

Use exactly this structure:

{{
  "cnNumber": "string or null",
  "hasSignature": true,
  "hasStamp": true,
  "hasHandwriting": true,
  "imageQualityPassed": true,
  "remarksText": "string or null",
  "deliveryDate": "YYYY-MM-DD or null",
  "physicalDamage": false,
  "businessDamage": false,
  "shortage": false
}}
"""

# --------------------------------------------------
# INPUT IMAGE
# --------------------------------------------------

image_path = "images/pod5.jpg"


# --------------------------------------------------
# PREPROCESS IMAGE
# --------------------------------------------------

processed_image_path = preprocess_image(image_path)


# --------------------------------------------------
# RUN OCR
# --------------------------------------------------

ocr_text = run_ocr(processed_image_path)


# --------------------------------------------------
# BUILD QWEN PROMPT
# --------------------------------------------------

prompt = build_prompt(ocr_text)


# --------------------------------------------------
# IMAGE → BASE64
# --------------------------------------------------

image_url = image_to_data_url(processed_image_path)


# --------------------------------------------------
# SEND IMAGE + OCR TO QWEN 3 VL 8B
# --------------------------------------------------

print("\n========== SENDING TO QWEN 3 VL 8B ==========\n")

response = client.chat.completions.create(

    model="qwen/qwen3-vl-8b-instruct",

    messages=[
        {
            "role": "user",
            "content": [

                {
                    "type": "text",
                    "text": prompt
                },

                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }

            ]
        }
    ]
)


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print("\n========== QWEN 3 VL 8B OUTPUT ==========\n")

print(response.choices[0].message.content)

print("\n===========================================\n")