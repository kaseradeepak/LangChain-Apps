from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

OLLAMA_BASE_URL="http://localhost:11434"
API_PATH = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen:1.8b"

def build_simple_chain_using_lcel():
    """
    ChatPromptTemplate -> ChatOllama -> StrOutputParser
    """

    # Build ChatPromptTemplate.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a beginner-friendly programming instructor.

Rules:
- Explain the concept clearly.
- Use exactly 3 bullet points.
- Each bullet point should be short.
- Do not add an introduction.
- Do not add a conclusion.
"""
            ),
            (
                "human",
                "Explain {topic} using a simple analogy from {analogy_domain}."
            ),
        ]
    )

    # Build the Model
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=1
    )

    # temperature controls creativity or randomness of the output generation.

    # StrOutputParser extracts text content from model outputs such as AIMessage or AIMessageChunk and converts it into a plain string.
    parser = StrOutputParser()

    # LCEL - Langchain Expression Language.
    chain = prompt | llm | parser

    return chain

chain = build_simple_chain_using_lcel()

result = chain.invoke(
    {
        "topic" : "LCEL in Langchain",
        "analogy_domain" : "sofware engineering"
    }
)

print(result)








