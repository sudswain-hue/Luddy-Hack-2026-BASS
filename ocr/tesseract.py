# ocr/tesseract.py
"""
Tesseract wrapper for full-document OCR.
"""
import pytesseract
from PIL import Image
from pathlib import Path


def extract_text_from_image(image_path: str) -> str:
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img  = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    text = " ".join(text.split())
    return text


def extract_text_from_pil(pil_image: Image.Image) -> str:
    text = pytesseract.image_to_string(pil_image)
    return " ".join(text.split())


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["tesseract", "--version"],
        capture_output=True, text=True
    )
    print(f"Tesseract version:\n{result.stdout.split(chr(10))[0]}")

    # paths match your actual folder structure: ocr/data/SimulatedNoisyOffice/
    test_candidates = [
        "data/SimulatedNoisyOffice/simulated_noisy_images_grayscale",
        "data/SimulatedNoisyOffice/clean_images_grayscale",
        "data/SimulatedNoisyOffice/clean_images_binaryscal (low resolution)",
        "data/SimulatedNoisyOffice/clean_images_grayscale_doubleresolution",      
    ]

    for candidate in test_candidates:
        p = Path(candidate)
        if p.is_dir():
            imgs = list(p.glob("*.png"))
            if imgs:
                test_image = imgs[0]
                print(f"\nTesting on: {test_image}")
                text = extract_text_from_image(str(test_image))
                print(f"\nExtracted length: {len(text)} chars")
                print(f"\nFull extracted text:\n{text}")
                break
    else:
        print("\nNo test image found — check folder path.")