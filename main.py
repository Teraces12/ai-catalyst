from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="AI Catalyst API",
    description="🧠 Summarize & Ask Questions from PDF files using LangChain + OpenAI",
    version="1.0.0"
)

@app.get("/", tags=["Health Check"])
def read_root():
    return JSONResponse(content={
        "status": "✅ API is running",
        "message": "Welcome to AI Catalyst - your smart PDF assistant!"
    })

