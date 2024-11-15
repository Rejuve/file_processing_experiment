import re
import spacy
### NER layer ###

# def process_text(text):
#     # Load the SciSpacy model
#     print(spacy.util.get_package_path("en_core_sci_md"))

#     import spacy


# import spacy


def process_text(text):
    # Load the SciSpaCy model
    nlp = spacy.load("en_core_sci_md")

    # Process the text extracted from OCR
    doc = nlp(text)
    # Dictionary to store test names and their associated values
    test_results = {}

    # Define a regular expression pattern to identify numeric values
    value_pattern = re.compile(r"\b\d+\.?\d*\b")

    # Loop through each entity in the document
    ent_group = []
    for ent in doc.ents:
        ent_group.append(ent.text)
        test_name = ent.text  # Entity name
        start_pos = (
            ent.end_char
        )  # Position after the entity to start searching for the value

        # Find the next numeric value in the text after the entity
        match = value_pattern.search(text, start_pos)
        if match:
            test_value = match.group()
            test_results[test_name] = test_value
        else:
            test_results[test_name] = "Value not found"
    print(f"\nEntity group : {ent_group}\n")
    return test_results
