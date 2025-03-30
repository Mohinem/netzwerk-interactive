from fastapi import APIRouter
from pydantic import BaseModel
from app.services.ai_service import ai_explain_grammar

router = APIRouter()

class GrammarQuery(BaseModel):
    topic: str

@router.post("/grammar")
def explain_grammar(query: GrammarQuery):
    explanation = ai_explain_grammar(query.topic)
    return {"explanation": explanation}
