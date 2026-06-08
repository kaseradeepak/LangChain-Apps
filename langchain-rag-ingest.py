from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

DATA_DIR=Path("/documents")

loader = DirectoryLoader(
    path=str(DATA_DIR),
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding" : "utf-8"}
)

docs = loader.load()

# 1. Created the documents.
# 2. Loading the documents.


