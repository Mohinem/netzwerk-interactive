from fastapi import APIRouter, UploadFile, File, HTTPException
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from pytesseract import Output
import io
import base64
import PyPDF2  # to get the number of pages

router = APIRouter()

@router.post("/digitize")
async def digitize_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    
    contents = await file.read()
    
    try:
        # Determine the total number of pages using PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        total_pages = len(pdf_reader.pages)
        
        structured_pages = []
        
        # Process one page at a time to reduce memory footprint.
        for page_num in range(1, total_pages + 1):
            # Convert only the current page (first_page and last_page are 1-indexed)
            pages = convert_from_bytes(
                contents, 
                dpi=300, 
                fmt='jpeg', 
                first_page=page_num, 
                last_page=page_num
            )
            # There should be exactly one image in the list
            page_image = pages[0]
            
            # Use Tesseract to extract OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(
                page_image,
                lang='eng',  # or 'deu' if needed
                config='--dpi 300 --psm 6 --oem 3',
                output_type=Output.DICT
            )
            
            # Gather recognized words with bounding boxes
            words = []
            for i in range(len(ocr_data['text'])):
                txt = ocr_data['text'][i].strip()
                try:
                    conf = int(ocr_data['conf'][i])
                except ValueError:
                    conf = 0
                if txt and conf > 0:
                    words.append({
                        "text": txt,
                        "left": ocr_data['left'][i],
                        "top": ocr_data['top'][i],
                        "width": ocr_data['width'][i],
                        "height": ocr_data['height'][i],
                    })
            
            # Convert the page image to Base64
            buffered = io.BytesIO()
            page_image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Get the original image dimensions
            width, height = page_image.size
            
            # Append the processed page to the result list
            structured_pages.append({
                "page_number": page_num,
                "width": width,
                "height": height,
                "words": words,
                "image": f"data:image/jpeg;base64,{image_base64}"
            })
            
            # Explicitly close the image to free up memory
            page_image.close()
        
        return {"pages": structured_pages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
