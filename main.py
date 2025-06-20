from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
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
    # Dummy response — replace with real LangChain logic
    return {"summary": f"This is a summary of {file.filename}."}

@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load and split PDF
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)

        # Embed and index with FAISS
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Search + Answer
        retriever = vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(question)

        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=relevant_docs, question=question)

        return {"answer": response}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
