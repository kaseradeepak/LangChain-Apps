# from langchain_openai import ChatOpenAI

# client = ChatOpenAI(
#     model="gpt-5.2"
# )

# response = client.invoke(
#     "Explain SQL indexes to the beginner students in simple words."
# )

# print(response.content)

# Let's try using PromptTemplate in the above API Call. 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

client = ChatOpenAI(
    model="gpt-5.2"
)

# Creates a reusable prompt template.
prompt_template = PromptTemplate.from_template(
    """
    Explain {topic} to {audience} students.

    Requirements:
    - Use {tone} tone.
    - Give one real-life analogy to explain the concept.
    - Keep the response within {limit} words.
    """
)

# Replaces variables with actual values provided by the user at runtime.
prompt = prompt_template.format(
    topic="LangChain Components",
    audience="Beginners",
    tone="simple",
    limit=1000
)

# Invokes LLM API to get the response.
response = client.invoke(prompt)

print(response)

# LCEL - LangChain Expression Language.
# LCEL allows us to connect components like a Pipeline.

# prompt -> LLM -> output parser.