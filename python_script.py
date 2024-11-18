import cv2
from PIL import Image
import pytesseract
import re
import json
import os
import logging
from spacy_model import process_text

def preprocess_image(image_path):
    """
    Preprocess an image by reading it in grayscale, applying thresholding to convert it to black and white,
    and returning the processed image as a PIL Image object.

    Args:
        image_path (str): The file path to the image to be processed.

    Returns:
        PIL.Image.Image: The processed black and white image.
    """
    # Read image with OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    print(f"Read image using OpenCV: {image}")
    # Apply thresholding to convert the image to black and white
    _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Resize or apply other filters if necessary
    return Image.fromarray(thresh_image)


def extract_text(image):
    """
    Extracts text from an image using Optical Character Recognition (OCR).

    Args:
      image: An image object that can be processed by pytesseract.

    Returns:
      str: The text extracted from the image.
    """
    # Configure pytesseract
    custom_config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(image, config=custom_config)
    return text


def parse_blood_report_text(text):
    """
    Parses a blood report text to extract biomarker names and their corresponding values.

    Args:
        text (str): The text content of the blood report.

    Returns:
        dict: A dictionary where the keys are biomarker names (formatted in lowercase with underscores) 
              and the values are the corresponding biomarker values as strings.
    """

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
            biomarker_name = match.group(1).strip()
            biomarker_name = biomarker_name.lower()
            biomarker_name = biomarker_name.replace(" ", "_")  # Format name
            value = match.group(2).replace(",", "")  # Remove commas from numbers

            # Store the name and value in the dictionary
            biomarkers[biomarker_name] = value

    return biomarkers


def process_lab_report(image_path):
    """
    Processes a lab report image and extracts relevant data.
    Args:
        image_path (str): The file path to the lab report image.
    Returns:
        dict: A dictionary containing the parsed data from the lab report.
    Steps:
        1. Preprocess the image.
        2. Extract text from the processed image.
        3. Parse the extracted text to obtain report data.
    """

    processed_image = preprocess_image(image_path)

    logging.basicConfig(level=logging.INFO)

    logging.basicConfig(level=logging.INFO)

    raw_text = extract_text(processed_image)
    logging.info(f"Extracted text: {raw_text}")

    report_data = process_text(raw_text)

    return report_data


def save_text_to_file(data, filename="simplified_lab_report"):
    """
    Save the extracted data to a JSON file.

    Args:
        data (dict): The data to be saved.
        filename (str): The name of the file to save the data to (default is "simplified_lab_report").

    Returns:
        None
    """
    with open('./extracted_data/'+filename + ".json", "w") as file:
        json.dump(data, file, indent=4)
    print(f"Result saved to {filename + '.json'}")


def process_all_reports(directory_path):
    """
    Processes all lab report images in the specified directory.

    This function iterates through all files in the given directory, identifies
    files with .jpg or .png extensions, processes each image to extract lab report
    data, and saves the extracted data to a text file.

    Args:
        directory_path (str): The path to the directory containing the lab report images.

    Returns:
        None
    """
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            report_data = process_lab_report(image_path)
            save_text_to_file(report_data, os.path.splitext(filename)[0])

process_all_reports("./sample_blood_test/")
