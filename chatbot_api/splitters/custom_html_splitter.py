import dotenv
import os
import uuid

from langchain.schema import Document
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

DOCUMENT_PATH = os.getenv("DOCUMENT_PATH")

def custom_html_splitter(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Extract headers and generate links
    chunks = []

    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        unique_id = str(uuid.uuid4())
        header["id"] = unique_id  # Assign unique ID
        header_link = f"# {unique_id}"  # Generate header link

        # Extract content for each section
        content = []
        for sibling in header.find_next_siblings():
            if sibling.name and sibling.name.startswith("h"):  # Stop at the next header
                break
            content.append(sibling.get_text(separator=" ", strip=True))  # Extract plain text

        # Combine the text content and clean it
        chunk_content = " ".join(content).strip()

        # Create a document chunk with metadata
        chunks.append(
            Document(
                page_content=chunk_content,
                metadata={
                    "header": header.get_text(separator=" ", strip=True),
                    "link": header_link
                }
            )
        )
    # The splitter to use to create smaller chunks
    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)

    doc_ids = [str(uuid.uuid4()) for _ in chunks]

    sub_docs = []
    for i, doc in enumerate(chunks):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata["doc_id"] = _id
        sub_docs.extend(_sub_docs)

    return sub_docs