from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import tempfile

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
