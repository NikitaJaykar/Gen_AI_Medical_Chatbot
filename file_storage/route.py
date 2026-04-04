from fastapi import FastAPI
from pydantic import BaseModel
from file_upload import vector_store   # adjust import if needed

app = FastAPI()

# Request body model
class Input(BaseModel):
    filename: str

@app.get("/hello")
def home():
    return {"message": "Hello"}

# POST route
@app.post("/fileupload")
def add_file(data: Input):
    vector_store(data.filename)
    return {"message": f"File '{data.filename}' processed successfully"}