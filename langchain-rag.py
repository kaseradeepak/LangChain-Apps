# Load the data from Chroma Db and create a retriever.
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR=Path("chroma_db")
COLLECTION_NAME="ecommerce_policy_docs"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)

# Create a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k" : 3}   
)

query = "What is the return window for electronics item?"

docs = retriever.invoke(query) # top-3 similar chunks from chroma db

print(f"User Query: {query}")
print(f"Retrieved documents: {len(docs)}")

