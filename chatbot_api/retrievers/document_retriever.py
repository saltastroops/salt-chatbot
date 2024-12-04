import dotenv
import logging
import os

from embeddings.custom_embeddings import MyEmbeddingFunction
from splitters.custom_html_splitter import custom_html_splitter
from tools.text_summaries import generate_text_summaries

from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document

dotenv.load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")
CHROMA_DATA_DIR = os.getenv("CHROMA_DATA_DIR")

DOCUMENT_PATH = os.getenv("DOCUMENT_PATH")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)

embeddings = MyEmbeddingFunction()

db = Chroma(
    collection_name=CHROMA_COLLECTION_NAME,
    persist_directory=CHROMA_DATA_DIR,
    embedding_function=embeddings
)

if len(db.get()["documents"]) > 0:
    retriever = db.as_retriever()
else:
    chunked_documents = custom_html_splitter(DOCUMENT_PATH)
    texts = [doc.page_content for doc in chunked_documents]
    doc_ids = [doc.metadata["doc_id"] for doc in chunked_documents]
    headers = [doc.metadata["header"] for doc in chunked_documents]
    links = [doc.metadata["link"] for doc in chunked_documents]

    LOGGER.info(msg="Generating summaries of the page content...")
    text_summaries, _ = generate_text_summaries(
        texts, [], summarize_texts=True
    )

    LOGGER.info(msg="Adding summarized documents to the vectorstore...")
    summary_docs = [
        Document(page_content=s, metadata={
            "doc_id": doc_ids[i],
            "header": headers[i],
            "link": links[i],
        })
        for i, s in enumerate(text_summaries)
    ]

    db = Chroma.from_documents(
        documents=summary_docs,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_DATA_DIR,
        embedding=embeddings
    )
    retriever = db.as_retriever()