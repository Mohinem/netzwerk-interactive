from fastapi import APIRouter, UploadFile, File, HTTPException
import pytesseract
from pdf2image import convert_from_bytes
from typing import List
import tempfile

router = APIRouter()

@router.post("/digitize")
async def digitize_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    contents = await file.read()

    try:
        extracted_text: List[str] = []
        
        # Use a temporary directory to manage memory better
        with tempfile.TemporaryDirectory() as path:
            pages = convert_from_bytes(contents, dpi=200, fmt='jpeg', output_folder=path, single_file=False)

            # Process PDF pages incrementally
            for page_number, page in enumerate(pages, start=1):
                text = pytesseract.image_to_string(page, lang='deu')
                extracted_text.append(f"--- Page {page_number} ---\n{text}")

                # Optional: Stop after a certain number of pages for quick tests
                # if page_number >= 10:
                #     break

        full_text = "\n".join(extracted_text)

        return {"extracted_text": full_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to digitize PDF: {str(e)}")
