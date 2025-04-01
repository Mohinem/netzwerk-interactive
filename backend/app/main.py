from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import exercises, auth, ocr, ai_assist
from app.db import database
from contextlib import asynccontextmanager



@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()

app = FastAPI(lifespan=lifespan)


# Enable CORS (allowing your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(exercises.router, prefix="/exercises", tags=["Exercises"])
app.include_router(ocr.router, prefix="/ocr", tags=["OCR"])
app.include_router(ai_assist.router, prefix="/ai", tags=["AI Assistance"])

@app.get("/")
def read_root():
    return {"message": "Netzwerk Backend Ready!"}
