from fastapi import FastAPI
from pydantic import BaseModel
from medical_chatbots import medical_chat   # remove dot if running directly


app = FastAPI()

# Request body model
class Input(BaseModel):
    query: str

@app.get("/hello")
def home():
    return {"message": "Hello"}

@app.post("/chat")
def chatbot(data: Input):
    result = medical_chat(data.query)
    return {"result": result}