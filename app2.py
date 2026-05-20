from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="gpt-5.2"
)

prompt = PromptTemplate.from_template(
    """
    Explain {topic} to {audience} students.

    Requirements:
    - Use {tone} tone.
    - Give one real-life analogy to explain the concept.
    - Keep the response within {limit} words.
    """
)

# Without StrOutputParser, model returns the complete response object.
# With StrOutputParser, we just get the content string from the response. 
output_parser = StrOutputParser()

# prompt -> llm -> output_parser
chain = prompt | llm | output_parser

# invoke / stream / batch
response = chain.invoke({
        "topic" : "SQL Indexes",
        "audience" : "Beginners",
        "tone" : "Simple",
        "limit" : 100
    })

print(response)



