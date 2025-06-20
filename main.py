from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI

from langdetect import detect
import tempfile
import traceback

app = FastAPI(
    title="AI Catalyst API",
    description="🧠 Summarize & Ask Questions from PDF files using LangChain + OpenAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict for production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
def read_root():
    return JSONResponse(content={
        "status": "✅ API is running",
        "message": "Welcome to AI Catalyst - your smart PDF assistant!"
    })


from langchain.chains.summarize import load_summarize_chain

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)

        if not docs:
            return JSONResponse(status_code=400, content={"error": "PDF contains no readable text."})

        llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

        # ✅ Use Map-Reduce to summarize large doc intelligently
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)

        return {"answer": summary}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()

        if not pages:
            return JSONResponse(status_code=400, content={"error": "No readable text found in PDF."})

        sample_text = pages[0].page_content[:200]
        lang = detect(sample_text)

        if lang != "en":
            return JSONResponse(
                status_code=400,
                content={"error": f"Non-English PDF detected (language: {lang}). Please upload English documents only."}
            )

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        retriever = vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(question)

        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=relevant_docs, question=question)

        return {"answer": str(response)}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
