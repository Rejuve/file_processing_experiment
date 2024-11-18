import re
import spacy
### NER layer ###

# def process_text(text):
#     # Load the SciSpacy model
#     print(spacy.util.get_package_path("en_core_sci_md"))

#     import spacy


# import spacy


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
