from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class Answer(BaseModel):
    answer: str

@router.get("/")
def get_exercise():
    return {
        "question": "Ich ____ gestern im Kino.",
        "options": ["bin", "war", "hat", "habe"],
        "correct": "war"
    }

@router.post("/")
def submit_answer(ans: Answer):
    correct = ans.answer == "war"
    next_question = {
        "question": "Wir ____ nach Berlin gefahren.",
        "options": ["sind", "haben", "ist", "seid"],
        "correct": "sind"
    }
    return {"correct": correct, "nextQuestion": next_question}
