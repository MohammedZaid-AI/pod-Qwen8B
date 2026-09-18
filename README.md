# POD OCR & Classification Pipeline

A Python-based Proof of Delivery (POD) document analysis pipeline. It
preprocesses a POD image, extracts text with PaddleOCR, sends the
processed image plus OCR text to a Qwen vision model through OpenRouter,
reconciles selected fields against OCR evidence, and prints a structured
classification as JSON.

> **Scope:** `ocr_test.py` processes **one image per run**. It is not
> the batch runner. The final classification is printed to the terminal
> as JSON; preprocessing images are saved in `processed_images/`.

## Pipeline overview

1.  Load one POD image.
2.  Apply EXIF orientation correction, attempt page-boundary detection
    and perspective correction, deskew small angles, and create an
    enhanced OCR image.
3.  With `POD_ROTATION_DEGREES=auto` (default), run OCR against 0°, 90°,
    180°, and 270° candidates and select a candidate using OCR
    text/confidence scoring.
4.  PaddleOCR extracts recognized text and bounding boxes.
5.  Send the corrected document image and OCR text to the configured
    Qwen vision model through OpenRouter. The model returns structured
    evidence as JSON.
6.  Reconcile and validate selected fields: check CN against OCR
    evidence, normalize and verify delivery dates against OCR, sanitize
    remarks, and apply Python classification rules.
7.  Print the final POD classification JSON.

## Requirements

-   Python version supported by your chosen PaddlePaddle/PaddleOCR
    releases. Python 3.10--3.12 is a practical starting point, but
    confirm compatibility for the versions you install.
-   OpenRouter account and API key.
-   Internet access for the OpenRouter API call.

Packages imported by the script: `paddlepaddle` (CPU), `paddleocr`,
`opencv-python`, `numpy`, `Pillow`, `python-dotenv`, and `requests`.

## Setup on Windows

Open PowerShell in the folder containing `ocr_test.py`.

Create and activate a virtual environment:

``` powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies:

``` powershell
python -m pip install paddlepaddle paddleocr opencv-python numpy Pillow python-dotenv requests
```

PaddleOCR and PaddlePaddle compatibility depends on their releases and
Python version. If installation fails, follow the official Paddle
installation guidance to select a compatible pair.

## Configure OpenRouter

Create `.env` beside `ocr_test.py`:

``` dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=qwen/qwen3-vl-8b-instruct
OPENROUTER_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_APP_NAME=POD Classification
# Optional:
# OPENROUTER_SITE_URL=https://your-project.example
```

Replace the placeholder key with your own. **Never share or commit
`.env`**; add it to `.gitignore`. The model must be available to your
OpenRouter account. API usage may require credits. HTTP 402 means the
request could not be funded at the requested size; check your balance or
adjust the request/model configuration.

## Add an input image

Example:

``` text
project/
├── ocr_test.py
├── .env
└── images/
    └── pod4.jpg
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`,
`.webp`.

By default, the script looks for `images/pod4.jpg`. To change it, add
this to `.env`:

``` dotenv
POD_IMAGE_PATH=images/your_pod_image.jpg
```

Use a path relative to the directory where you launch Python, or an
absolute path.

## Run

``` powershell
python ocr_test.py
```

The terminal displays preprocessing messages, OCR text, structured Qwen
evidence, and the **FINAL POD CLASSIFICATION JSON**.

### Rotation

Default:

``` dotenv
POD_ROTATION_DEGREES=auto
```

Or set a known fixed rotation:

``` dotenv
POD_ROTATION_DEGREES=0
# accepted values: 0, 90, 180, 270
```

A fixed value is applied before preprocessing. In auto mode, the script
compares OCR results for four right-angle rotations. The selected angle
is based on OCR confidence and recognized character count; this is a
heuristic and can choose incorrectly on difficult documents.

## Files created

The script creates `processed_images/` in the current working directory
and saves:

-   `<image-name>_document.jpg` --- rectified/deskewed document sent to
    the vision model.
-   `<image-name>_ocr_enhanced.png` --- enhanced image used for OCR.

The final classification JSON is printed to the terminal. **This script
does not automatically save a final JSON or CSV file.** Copy the printed
JSON if you need to store it, or add a separate output-writing step.

## Final JSON fields

The output includes fields such as `cnNumber`, `hasSignature`,
`hasStamp`, `hasHandwriting`, `imageQualityPassed`, `remarksText`,
`deliveryDate`, `categoryReason`, `confidenceScore`, `podCategory`, and
`limit_exceed`.

### Categories

-   `MANUAL_CHECK_REQUIRED`
-   `ISSUE_POD_DAMAGED_AND_SHORT`
-   `ISSUE_POD_DAMAGED`
-   `ISSUE_POD_SHORT`
-   `CLEAN_POD_SEAL_AND_SIGNATURE`
-   `CLEAN_POD_ONLY_SEAL`
-   `CLEAN_POD_ONLY_SIGNATURE`
-   `NO_SIGNATURE_NO_STAMP`

The Python classifier prioritizes image-quality and
physical-paper-damage review, then goods-damage/shortage evidence, and
then signature/stamp presence. Manual review is an uncertainty flag, not
necessarily a software crash.

## Troubleshooting

-   **`OPENROUTER_API_KEY is missing`:** Check that `.env` is beside the
    script and the key is set.
-   **OpenRouter HTTP 401:** Verify the API key and account access.
-   **OpenRouter HTTP 402:** Check available credits; the request may
    exceed the available balance.
-   **Paddle installation errors:** Confirm
    Python/PaddleOCR/PaddlePaddle compatibility and use a fresh virtual
    environment if needed.
-   **`POD image not found`:** Check `POD_IMAGE_PATH`, filename, and
    working directory.
-   **`MANUAL_CHECK_REQUIRED`:** Review `categoryReason` and the
    structured evidence printed earlier. The pipeline flags ambiguous
    important fields, poor image quality, or uncertain CN values for
    review.

## Notes

-   This is an OCR + vision-model pipeline, **not a custom-trained
    model**.
-   Results depend on image quality, OCR output, model response, API
    availability, and validation/classification rules.
-   Manually review consequential POD decisions; model output is not
    infallible.
-   Do not upload confidential POD documents to external services unless
    authorized.
-   `ocr_test.py` is for one image per run. Batch processing requires a
    separate runner and incremental result saving.
