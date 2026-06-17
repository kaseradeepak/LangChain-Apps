#Step-2
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import shutil

DATA_DIR=Path("documents")
CHROMA_DIR=Path("chroma_db")
COLLECTION_NAME="ecommerce_policy_docs"

# DirectoryLoader loads multiple files from the given path
# TextLoader reads the text data from the files.
loader = DirectoryLoader(
    path=str(DATA_DIR),
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding" : "utf-8"}
)

docs = loader.load()

# Split the documents into Chunks.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    add_start_index=True
)

chunks = text_splitter.split_documents(docs)

print(f"Original documents: {len(docs)}")
print(f"Generated chunks: {len(chunks)}")

# Try to print the chunks

# Store these chunks into vector db.

# Clean old db - optional step.
if CHROMA_DIR.exists():
    shutil.rmtree(CHROMA_DIR)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)

ids = vector_store.add_documents(documents=chunks)

# persist_directory=str(CHROMA_DIR) -> This tell Chroma DB to store vectors locally, so that we can reuse them later.

# Load Documents -> Split into chunks -> Create embeddings -> Store embeddings into Chroma DB






