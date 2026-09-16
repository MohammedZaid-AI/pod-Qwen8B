# ============================================================
# POD OCR TEST
# PaddleOCR + Image Preprocessing
# Windows CPU configuration
# ============================================================

import os


# ============================================================
# PADDLE CONFIGURATION
# ============================================================

# Disable oneDNN / MKLDNN
os.environ["FLAGS_use_mkldnn"] = "0"

# Disable Paddle PIR API
os.environ["FLAGS_enable_pir_api"] = "0"

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Limit CPU threads
os.environ["OMP_NUM_THREADS"] = "4"


# ============================================================
# IMPORTS
# ============================================================

import paddle

from PIL import Image, ImageEnhance, ImageOps
from paddleocr import PaddleOCR


# Explicitly disable MKLDNN
paddle.set_flags({
    "FLAGS_use_mkldnn": False
})


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    print("\n========== PREPROCESSING ==========\n")

    # Load image
    image = Image.open(image_path)

    print(f"Original image size: {image.size}")

    # --------------------------------------------------------
    # Fix EXIF orientation
    # --------------------------------------------------------

    image = ImageOps.exif_transpose(image)

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    image = image.convert("RGB")

    # --------------------------------------------------------
    # Improve contrast
    # --------------------------------------------------------

    image = ImageEnhance.Contrast(
        image
    ).enhance(1.2)

    # --------------------------------------------------------
    # Improve sharpness
    # --------------------------------------------------------

    image = ImageEnhance.Sharpness(
        image
    ).enhance(1.3)

    # --------------------------------------------------------
    # Save processed image
    # --------------------------------------------------------

    os.makedirs(
        "processed_images",
        exist_ok=True
    )

    filename = os.path.basename(
        image_path
    )

    output_path = os.path.join(
        "processed_images",
        filename
    )

    image.save(
        output_path,
        quality=95
    )

    print(
        f"Processed image saved to: {output_path}"
    )

    return output_path


# ============================================================
# OCR MODEL
# ============================================================

def run_ocr(image_path):

    print(
        "\n========== LOADING OCR MODEL ==========\n"
    )

    ocr = PaddleOCR(
        lang="en",
        device="cpu"
    )

    print(
        "\n========== RUNNING OCR ==========\n"
    )

    result = ocr.predict(
        image_path
    )

    return result


# ============================================================
# EXTRACT CLEAN OCR TEXT
# ============================================================

def extract_ocr_text(result):

    """
    Extract only recognized text from PaddleOCR.

    Removes:
    - bounding boxes
    - polygons
    - model configuration
    - other raw OCR metadata

    Keeps:
    - recognized text
    - reasonably confident detections
    """

    all_text = []

    for page in result:

        # PaddleOCR 3.x returns a result object
        if not hasattr(page, "json"):
            continue

        data = page.json

        # Some versions expose json as a callable
        if callable(data):
            data = data()

        # Get result dictionary
        res = data.get(
            "res",
            {}
        )

        # Recognized text
        texts = res.get(
            "rec_texts",
            []
        )

        # Recognition confidence
        scores = res.get(
            "rec_scores",
            []
        )

        # ----------------------------------------------------
        # Extract text + confidence
        # ----------------------------------------------------

        for text, score in zip(
            texts,
            scores
        ):

            text = str(text).strip()

            # Ignore empty detections
            if not text:
                continue

            # Ignore very low-confidence detections
            if score < 0.50:
                continue

            all_text.append(text)

    return "\n".join(all_text)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Input POD
    # --------------------------------------------------------

    image_path = "images/pod2.jpg"

    # --------------------------------------------------------
    # Check image exists
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        print(
            f"\nERROR: Image not found: {image_path}"
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # STEP 1
    # Preprocess image
    # --------------------------------------------------------

    processed_image = preprocess_image(
        image_path
    )

    # --------------------------------------------------------
    # STEP 2
    # Run OCR
    # --------------------------------------------------------

    ocr_result = run_ocr(
        processed_image
    )

    # --------------------------------------------------------
    # STEP 3
    # Extract clean text
    # --------------------------------------------------------

    ocr_text = extract_ocr_text(
        ocr_result
    )

    # --------------------------------------------------------
    # STEP 4
    # Display clean OCR
    # --------------------------------------------------------

    print(
        "\n========== EXTRACTED OCR TEXT ==========\n"
    )

    if ocr_text:

        print(ocr_text)

    else:

        print(
            "No text was detected."
        )

    print(
        "\n=========================================\n"
    )