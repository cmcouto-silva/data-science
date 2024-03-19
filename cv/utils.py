import fitz  # PyMuPDF
import PyPDF2
from pdf2image import convert_from_path

def rasterize_pdf(pdf_path, img_base_path, zoom=2):
    """Convert PDF to image files using PyMuPDF"""
    doc = fitz.open(pdf_path)
  
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        output_img_path = f"{img_base_path}_page{page_num + 1}.png"
        pix.save(output_img_path)

    doc.close()

def count_pdf_pages(pdf_path):
    """
    Count the number of pages in a PDF file.

    Parameters:
    - pdf_path: Path to the PDF file.

    Returns:
    - The number of pages in the PDF.
    """
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        return len(pdf_reader.pages)
