import logging
import os
import tempfile
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from generate import generate_text as generate_answer
from retrieve import hybrid_search, index_documents

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS - allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        doc_id = str(uuid4())

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "upload.pdf")
            with open(tmp_path, "wb") as tmp:
                tmp.write(file.file.read())
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No documents found in the uploaded file.",
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the uploaded file.",
            )

        logging.info("Chunks created: %s", len(chunks))
        index_documents(chunks, doc_id)

        return {
            "message": "Indexed successfully",
            "doc_id": doc_id,
            "chunks": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ask")
def ask(query: str, doc_id: str):
    try:
        if not query.strip() or not doc_id.strip():
            raise HTTPException(
                status_code=400,
                detail="Missing query or doc_id",
            )

        logging.info("Query: %s", query)
        logging.info("Doc ID: %s", doc_id)

        docs = hybrid_search(query, doc_id)
        logging.info("Retrieved %s chunks", len(docs))
        for i, d in enumerate(docs):
            logging.info("Chunk %s: %s...", i, d[:100])

        if not docs:
            return {
                "answer": "No relevant documents found.",
                "docs": [],
            }

        answer = generate_answer(query, docs)
        return {
            "answer": answer,
            "docs": docs,
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Ask failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
