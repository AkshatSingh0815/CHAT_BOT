import fitz
import easyocr
import numpy as np
from pdf2image import convert_from_path
import cv2
from PIL import Image
import io
reader = easyocr.Reader(['en'], gpu=False)
def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    return thresh
def ocr_image(image):
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        image_np = preprocess_image(image_np)
    results = reader.readtext(image_np)
    text = []
    for r in results:
        text.append(str(r) if len(r) > 1 else "")
    return " ".join(text)

def extract_scanned_pages(pdf_path):
    extracted_text = []
    # Use a generator to yield one image at a time
    for page in convert_from_path(pdf_path, thread_count=1):
        text = ocr_image(page)
        if text.strip():
            extracted_text.append(text)
        # Manually clear the image object to free RAM
        page.close() 
    return "\n".join(extracted_text)

def extract_embedded_images(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_text = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"]))
            
            text = ocr_image(image)
            if text.strip():
                extracted_text.append(text)
            
            # Help Python's Garbage Collector
            image.close()
    doc.close()
    return "\n".join(extracted_text)
def extract_all_image_text(pdf_path):

    scanned = extract_scanned_pages(pdf_path)

    embedded = extract_embedded_images(pdf_path)

    return scanned + "\n" + embedded
