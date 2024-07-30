import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
import json
import cv2
import numpy as np
from PIL import Image
import pytesseract
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        num_pages = len(pdf.pages)
        st.sidebar.write(f"Total Pages: {num_pages}")
        page_number = st.sidebar.number_input("Select Page Number", min_value=1, max_value=num_pages, value=1)
        page = pdf.pages[page_number - 1]
        text = page.extract_text()
        
        # Extract form fields
        reader = PdfReader(pdf_file)
        form_fields = []

        if '/AcroForm' in reader.trailer['/Root']:
            form = reader.trailer['/Root']['/AcroForm']
            fields = form['/Fields']
            
            for field in fields:
                field_object = field.get_object()
                field_name = field_object.get('/T')  
                field_type = field_object.get('/FT')
                field_rect = field_object.get('/Rect')  
                if field_rect:
                    field_rect = [float(coord) for coord in field_rect]

                # Check field type to render input accordingly
                if field_type == '/Tx':  # Text field
                    st.sidebar.text_input(field_name, key=field_name)
                elif field_type == '/Btn':  # Checkbox
                    st.sidebar.checkbox(field_name, key=field_name)

                form_fields.append({
                    'name': field_name,
                    'type': field_type,
                    'rect': field_rect
                })

        return text, form_fields, page, page_number

def visualize_pdf_page(page):
    page_image = page.to_image()
    st.write("Visualizing the PDF page:")
    st.image(page_image.original, caption=f"PDF Page")  

def associate_titles_with_fields(json_data, fields):
    associations = []
    for field in fields:
        field_name = field.get('name')
        field_type = field.get('type')
        field_rect = field.get('rect')
        
        for title in json_data.get('titles', []):
            title_text = title.get('text')
            title_rect = title.get('rect')
            
            if field_name and title_text and field_rect and is_close(field_rect, title_rect):
                field_info = {
                    'title': title_text,
                    'field_name': field_name,
                    'field_type': field_type,
                    'field_rect': field_rect
                }
                associations.append(field_info)
    return associations

def is_close(rect1, rect2, threshold=50):
    x1, y1, x1_w, y1_h = rect1
    x2, y2, x2_w, y2_h = rect2
    center1 = ((x1 + x1_w) / 2, (y1 + y1_h) / 2)
    center2 = ((x2 + x2_w) / 2, (y2 + y2_h) / 2)
    return np.linalg.norm(np.array(center1) - np.array(center2)) < threshold

def process_image(image_file):
    uploaded_image = np.array(Image.open(image_file)) 
    gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(morph, 50, 150)
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel_dilate, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fields_detected = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h)
        if 500 < area < 50000 and 0.1 < aspect_ratio < 10:
            fields_detected.append({
                'position': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio
            })
    try:
        ocr_text = pytesseract.image_to_string(Image.fromarray(gray))
    except pytesseract.TesseractNotFoundError as e:
        st.error(f"Error: {e}. Make sure Tesseract OCR is installed and set up correctly.")
        ocr_text = ""
    
    return fields_detected, ocr_text, uploaded_image

st.title("Enhanced PDF and Image Form Processor")

st.sidebar.header("Upload your PDF and JSON")
pdf_file = st.sidebar.file_uploader("Upload PDF Form", type=["pdf"])
json_file = st.sidebar.file_uploader("Upload JSON File", type=["json"])
associations = []

if pdf_file and json_file:
    pdf_text, fields, pdf_page, page_number = process_pdf(pdf_file)
    json_data = json.load(json_file)
    
    st.subheader(f"Processing PDF Form Page {page_number}")
    st.write("Extracted Text from PDF:")
    st.text(pdf_text)

    st.write("Form Fields in PDF:")
    st.json(fields)

    visualize_pdf_page(pdf_page)

    associations = associate_titles_with_fields(json_data, fields)

st.sidebar.header("Upload your Image")
image_file = st.sidebar.file_uploader("Upload Image of PDF Form", type=["jpg", "jpeg", "png"])

if image_file:
    fields_detected, ocr_text, uploaded_image = process_image(image_file)
    
    st.subheader("Detected Fields in Image")
    st.image(image_file, caption='Uploaded Image', use_column_width=True)
    st.write("OCR Extracted Text:")
    st.text(ocr_text)
    st.json(fields_detected)
    image_with_fields = uploaded_image.copy()  
    for field in fields_detected:
        x, y, w, h = field['position']
        cv2.rectangle(image_with_fields, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    st.image(image_with_fields, caption='Detected Fields Highlighted', use_column_width=True)

if associations:
    st.download_button(
        label="Download Field Associations as JSON",
        data=json.dumps(associations, indent=4),
        file_name='field_associations.json',
        mime='application/json'
    )
