from fastapi import FastAPI, UploadFile, File, Query
from ingest import ingest_textbook
from rag import ask_question
import shutil, os

app = FastAPI()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks, index = ingest_textbook(path)
    return {"message": "PDF processed successfully", "chunks": len(chunks)}

@app.get("/ask")
async def ask(question: str = Query(..., description="Your question about the PDF")):
    answer = ask_question(question)
    return {"question": question, "answer": answer}
