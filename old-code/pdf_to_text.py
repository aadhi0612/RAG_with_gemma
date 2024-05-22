import fitz  # PyMuPDF
import os
import pathlib
import pytesseract
from PIL import Image

def extract_text_from_pdf(pdf_file):
    """
    Extract text (including OCR for images) from a PDF file.

    Args:
        pdf_file (str): Path to the input PDF file.

    Returns:
        str: Extracted text content from the PDF.
    """
    pdf_document = fitz.open(pdf_file)
    text_content = ""

    for page in pdf_document:
        # Extract text from the page
        page_text = page.get_text()

        # If the page text is empty, perform OCR on images
        if not page_text:
            images = page.get_images(full=True)
            for img_index, img_info in enumerate(images):
                xref = img_info[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.frombytes("RGB", (base_image["width"], base_image["height"]), image_bytes)
                image_text = pytesseract.image_to_string(image)
                text_content += image_text + "\n"

        else:
            text_content += page_text + "\n"

    pdf_document.close()
    return text_content.strip()

def convert_pdf_to_txt_with_ocr(input_folder, output_folder):
    """
    Convert PDF files in the input folder to TXT files in the output folder, applying OCR when needed.

    Args:
        input_folder (str): Path to the folder containing input PDF files.
        output_folder (str): Path to the folder for saving output TXT files.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pdf"):
            pdf_file = os.path.join(input_folder, file_name)
            txt_file = pathlib.Path(output_folder) / pathlib.Path(file_name).with_suffix(".txt")

            if txt_file.exists():
                print(f"TXT file already exists for '{file_name}', skipping conversion.")
                continue

            try:
                # Extract text content from the PDF (including OCR for images)
                text_content = extract_text_from_pdf(pdf_file)

                # Write extracted text to the output TXT file
                with open(txt_file, "w", encoding="utf-8") as txt:
                    txt.write(text_content)

                print(f"Successfully converted PDF '{file_name}' to TXT '{txt_file.name}'.")

            except Exception as e:
                print(f"Error encountered during PDF to TXT conversion for '{file_name}': {str(e)}")

# Usage example:
if __name__ == "__main__":
    
    input_folder = "/home/ubuntu/RAG/datasets/PDFs"
    output_folder = "/home/ubuntu/RAG/datasets/TXTs"
    
    
    convert_pdf_to_txt_with_ocr(input_folder, output_folder)

