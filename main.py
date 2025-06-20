from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Catalyst API",
    description="🧠 Summarize & Ask Questions from PDF files using LangChain + OpenAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Make more strict for production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
def read_root():
    return JSONResponse(content={
        "status": "✅ API is running",
        "message": "Welcome to AI Catalyst - your smart PDF assistant!"
    })

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    # Dummy response — replace with real LangChain logic
    return {"summary": f"This is a summary of {file.filename}."}

@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    # Dummy response — replace with real logic
    return {"answer": f"You asked: '{question}' based on {file.filename}."}
