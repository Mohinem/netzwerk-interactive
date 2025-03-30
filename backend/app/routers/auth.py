from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.config import SECRET_KEY, ALGORITHM
from jose import jwt
from passlib.context import CryptContext

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"])

fake_users_db = {"user": {"username": "user", "hashed_password": pwd_context.hash("pass")}}

class Login(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(data: Login):
    user = fake_users_db.get(data.username)
    if not user or not pwd_context.verify(data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = jwt.encode({"sub": user["username"]}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token}
