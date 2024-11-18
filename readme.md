# Blood Report OCR and Analysis

This project processes lab report images(PNG/JPG) to extract and analyze blood test data using Optical Character Recognition (OCR) and Natural Language Processing (NLP).

## Project Structure
```
/BloodReportOCR
|-- /extracted_data: Contains the extracted text data from the blood report images.
|-- /sample_blood_test: Contains sample blood test images used for testing and development.
|-- requirements.txt: Lists all the dependencies required to run the project.
|-- README.md: The project documentation file.
|-- spacy_model.py: Script for processing the extracted text using spaCy for NLP tasks.
|-- python_script.py: Main script to handle the OCR and analysis process.
```

## Requirements

The project requires the following Python packages:

- opencv-python: For image processing and manipulation.
- Pillow: For image handling and manipulation.
- pytesseract: For Optical Character Recognition (OCR) to extract text from images.
- spacy: For Natural Language Processing (NLP) to analyze and parse the extracted text.

You can install the required packages using the following command:

```sh
pip install -r [requirements.txt]
```

 
## Steps to perform image analysis

There are some steps I have followed to perform the extraction of biomarkers from the images.

### Image preprocessing

- Read the image in grayscale using OpenCV to reduce complexity and improve OCR accuracy.
- Apply thresholding (using Otsu's method) to convert the image to black and white, enhancing text regions.
- Optionally, resize or apply additional filters if necessary to further improve the image quality.

```python
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
```
### Text extraction using OCR

- Use the `pytesseract` library to extract text from the preprocessed image.
- Configure `pytesseract` to optimize text extraction based on the image characteristics. This can include setting the OCR engine mode (OEM) and page segmentation mode (PSM) to improve accuracy.

```python
def extract_text(image):
  """
  Extracts text from an image using Optical Character Recognition (OCR).

  Args:
    image: An image object that can be processed by pytesseract.

  Returns:
    str: The text extracted from the image.
  """
  # Configure pytesseract
  custom_config = r'--oem 3 --psm 6'
  text = pytesseract.image_to_string(image, config=custom_config)
  return text
```

### Text analysis
- Here I used two different methods for the analysis. 

#### First -> Regular Expression method
- I used a regular expression to extract health biomarkers from the extracted text.

```python
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
```
- issue with the above approach is 
  - 1, Not able to address different image format
  - 2, Not able to differenciate value and specfic medical biomarkers it works with alphabet, numberic and special characters so it doesn't have a capability of differenciating a medical term.
  #### Second -> Using NLP 
  - Load the extracted text into the `spacy` NLP model.
  - Use `spacy` to tokenize the text and identify relevant entities such as biomarkers and their values.
  - Parse the text to extract structured information for further analysis.

  ```python
  def process_text(text):
    """
    Processes the given text using the SciSpaCy model to extract test names and their associated values.

    Args:
      text (str): The text to be processed, typically extracted from OCR.

    Returns:
      dict: A dictionary where the keys are test names (entities) and the values are the associated numeric values 
          found in the text. If a value is not found for a test name, the value will be "Value not found".

    Example:
      >>> text = "Hemoglobin: 13.5\nWhite Blood Cell Count 4500"
      >>> process_text(text)
      {'Hemoglobin': '13.5', 'White Blood Cell Count': '4500'}
    """
    # Load the SciSpaCy model
    nlp = spacy.load("en_core_sci_md")
    # Ensure the model is loaded correctly
    if not nlp:
      raise ValueError("Failed to load the SciSpaCy model 'en_core_sci_md'")

    # Process the text extracted from OCR
    doc = nlp(text)
    # Dictionary to store test names and their associated values
    test_results = {}

    # Loop through each entity in the document
    for ent in doc.ents:
      test_name = ent.text  # Entity name
      test_value = ent.label_  # Entity label
      test_results[test_name] = test_value

    return test_results
  ```

### Data storage and visualization

- Store the extracted and analyzed data in a structured format (e.g., JSON or CSV).

```python
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
```

- Use visualization tools to present the data in a meaningful way, such as charts or graphs.


