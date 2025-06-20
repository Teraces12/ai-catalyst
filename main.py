from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langdetect import detect

import tempfile
import traceback
import os

# Load environment variables securely
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="AI Catalyst API",
    description="🧠 Cognitive PDF Assistant using LangChain + OpenAI",
    version="2.2.0 - Ultra Modern UI"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
def read_root():
    return JSONResponse(content={
        "status": "✅ API is running",
        "message": "Welcome to AI Catalyst - your ultra-modern PDF assistant!",
        "logo_url": "/static/logo.png",
        "ui_theme": "ultra-modern",
        "animations": True
    })

def get_llm(model_name="gpt-3.5-turbo-16k", temperature=0):
    return ChatOpenAI(model_name=model_name, temperature=temperature)

@app.post("/summarize")
async def summarize_pdf(
    file: UploadFile = File(...),
    model_name: str = Form("gpt-3.5-turbo-16k"),
    temperature: float = Form(0.0),
    allow_non_english: bool = Form(False),
    start_page: int = Form(1),
    end_page: int = Form(10000)
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        all_pages = loader.load()
        pages = all_pages[start_page - 1:end_page]

        sample_text = pages[0].page_content[:300] if pages else ""
        lang = detect(sample_text) if sample_text else "unknown"

        if lang != "en" and not allow_non_english:
            return JSONResponse(status_code=400, content={"error": f"Detected non-English text: {lang}"})

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        llm = get_llm(model_name=model_name, temperature=temperature)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)

        citations = [d.metadata.get("source", "page") for d in docs[:3]]

        return {
            "answer": summary,
            "language": lang,
            "citations": citations,
            "animated": True,
            "theme": "ultra-modern"
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask")
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...),
    model_name: str = Form("gpt-3.5-turbo-16k"),
    temperature: float = Form(0.0),
    allow_non_english: bool = Form(False),
    start_page: int = Form(1),
    end_page: int = Form(10000)
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        all_pages = loader.load()
        pages = all_pages[start_page - 1:end_page]

        sample_text = pages[0].page_content[:300] if pages else ""
        lang = detect(sample_text) if sample_text else "unknown"

        if lang != "en" and not allow_non_english:
            return JSONResponse(status_code=400, content={"error": f"Detected non-English text: {lang}"})

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(question)

        llm = get_llm(model_name=model_name, temperature=temperature)
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=relevant_docs, question=question)

        citations = [doc.metadata.get("source", "page") for doc in relevant_docs[:3]]

        return {
            "answer": answer,
            "language": lang,
            "citations": citations,
            "animated": True,
            "theme": "ultra-modern"
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
