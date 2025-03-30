from fastapi import FastAPI
from app.routers import exercises, auth, ocr, ai_assist
from app.db import database

app = FastAPI()

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(exercises.router, prefix="/exercises", tags=["Exercises"])
app.include_router(ocr.router, prefix="/ocr", tags=["OCR"])
app.include_router(ai_assist.router, prefix="/ai", tags=["AI Assistance"])

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.get("/")
def read_root():
    return {"message": "Netzwerk Backend Ready!"}
