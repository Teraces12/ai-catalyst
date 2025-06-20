from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import tempfile

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
    try:
        # Save uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load and split PDF
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)

        # Generate summary using LangChain LLM chain
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        summary = chain.run(input_documents=docs, question="Summarize this document in 5 sentences.")

        return {"summary": summary}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
