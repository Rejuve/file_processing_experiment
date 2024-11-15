import cv2
from PIL import Image
import pytesseract
import re
from spacy_model import process_text

def preprocess_image(image_path):
    # Read image with OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    print(f"Read image using OpenCV: {image}")
    # Apply thresholding to convert the image to black and white
    _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Resize or apply other filters if necessary
    return Image.fromarray(thresh_image)


def extract_text(image):
    # Perform OCR on the image
    text = pytesseract.image_to_string(image)
    return text


def parse_blood_report(text):
    # Example parsing for values (e.g., Hemoglobin, WBC, etc.)

    # Define a regex pattern to capture biomarker names and values
    pattern = re.compile(
        r"([A-Za-z\s\-\(\)%]+):?\s+([\d,]+\.\d+|\d+)"  # Biomarker name and main value
    )

    # Dictionary to store the biomarker data
    biomarkers = {}

    # Apply regex to each line in the text
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            # Extract biomarker name and value
            biomarker_name = match.group(1).strip().lower().replace(" ", "_")  # Format name
            value = match.group(2).replace(",", "")  # Remove commas from numbers

            # Store the name and value in the dictionary
            biomarkers[biomarker_name] = value

    return biomarkers


def process_lab_report(image_path):
    # Step 1: Preprocess the image
    processed_image = preprocess_image(image_path)

    print(f"Pre processed image: {processed_image}")
    # Step 2: Extract text with Tesseract
    raw_text = extract_text(processed_image)
    print(f"Extracted text: {raw_text}")

    # Step 3: Parse the extracted text to structure data
    report_data = process_text(raw_text)

    return report_data
    # return raw_text


import json


def save_text_to_file(data, filename="simplified_lab_report"):
    # Use json.dump() to write the dictionary with indentation
    with open('./extracted_data/'+filename + ".json", "w") as file:
        json.dump(data, file, indent=4)
    print(f"Result saved to {filename + '.json'}")


report = save_text_to_file(
    process_lab_report("./sample_blood_test/sample_blood_report_5.jpg"),
    "sample_blood_report_5",
)
print(report)
