from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

raw_documents = [
    Document(
        page_content="""
        Refund Policy:
        Learners can request a full refund within 7 days of enrollment if they have attended
        fewer than 2 live classes. After 7 days, refund requests are reviewed case-by-case.
        No refund is guaranteed after 30 days of enrollment.
        """,
        metadata={"source": "refund_policy"}
    ),
    Document(
        page_content="""
        Attendance Policy:
        Learners must maintain at least 75% attendance to remain eligible for career services.
        If attendance falls below 75%, the learner may be asked to complete make-up assignments.
        """,
        metadata={"source": "attendance_policy"}
    ),
    Document(
        page_content="""
        Project Submission Policy:
        Backend and system design projects must be submitted before the announced deadline.
        A learner may request one extension of up to 7 days if there is a valid reason.
        Repeated delays may affect certificate and placement eligibility.
        """,
        metadata={"source": "project_policy"}
    ),
    Document(
        page_content="""
        Batch Change Policy:
        A learner can request a batch change only once during the program.
        Batch change approval depends on seat availability, attendance record, and mentor approval.
        """,
        metadata={"source": "batch_policy"}
    ),
    Document(
        page_content="""
        Career Services Policy:
        Learners become eligible for placement support after completing mandatory projects,
        mock interviews, resume review, and maintaining minimum attendance requirements.
        """,
        metadata={"source": "career_services_policy"}
    ),
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

split_documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(
    documents=split_documents,
    embedding=embeddings,
    collection_name="course_policy_correction"
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k" : 3}
)

# retrieval tool agent
course_policy_tool = create_retriever_tool(
    retriever=retriever,
    name="course_policy_tool",
    description=(
        "Searches official course policy documents for refund rules, attendance requirements, "
        "batch change policy, project submission deadlines, placement eligibility, and extension rules. "
        "Use this tool only for questions about course policies or learner program rules."
    )
)

# non retrieval tool agent
@tool
def get_ticket_status(ticket_id: str):
    """
    Returns support ticket status for a given ticket id.
    Use this tool when the user asks about ticket status, refund request status,
    support request status, escalation status, or ticket tracking.
    """

    fake_ticket_database = {
        "TKT-1001": "Your refund request is under review. Expected response time: 2 working days.",
        "TKT-1002": "Your batch change request has been approved. New batch starts next Monday.",
        "TKT-1003": "Your project extension request was rejected because one extension was already used.",
        "TKT-1004": "Your career services eligibility review is pending mentor approval.",
    }

    ticket_status = fake_ticket_database.get(ticket_id)

    if not ticket_status:
        return f"Ticket Status not found for the given ticked id: {ticket_id}" 

    return ticket_status

# 2 tools
tools = [course_policy_tool, get_ticket_status]

llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
    (
        "system",
        """
        You are a helpful course support assistant.

        You have access to two tools:

        1. course_policy_search:
           Use this for questions about refund policy, attendance rules, batch change,
           project submission, extension, certificate, and career services eligibility.

        2. get_ticket_status:
           Use this when the user gives a ticket id or asks about support ticket status.

        Rules:
        - If the question is about official course policy, use course_policy_search.
        - If the question is about ticket status, use get_ticket_status.
        - If the question needs both policy and ticket status, use both tools.
        - If the question is outside course support, politely say that you can only help with course policies and support tickets.
        - Do not invent policy details.
        - When answering from documents, mention that the answer is based on the available policy documents.
        - Keep answers clear and student-friendly.
        """
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "{input}"),

    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3
)

chat_history = []

def ask_agent(user_query: str) -> str:
    """
    Sends the user query to the agent and stores conversation history in chat history manually.
    """

    response = agent_executor.invoke(
        {
            "input" : user_query,
            "chat_history" : chat_history
        }
    )

    answer = response["output"]

    # append human message and AI message in the chat_history.
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=answer))

    return answer

# query1 = "What is refund policy ?"
# print("User Query: ", query1)
# print("Response: ", ask_agent(query1))

# query2 = "What is the status of ticket TKT-1001?"
# print("User Query: ", query2)
# print("Response: ", ask_agent(query2))

# query3 = "What is the status of my TKT-1002 ? And tell me the refund policy."
# print("User Query: ", query3)
# print("Response: ", ask_agent(query3))

# Documents
# Chunking
# Embeddings 
# Storing embeddings into Vector Database.
# Retrieval Tool - course policy tool
# Get Ticket Status tool.
# prompt template.
# llm
# agent 
# agent_executor

# agent_scratchpad : Maintains information regarding tool calling.
# Without agent_scratchpad, the tool calling agent won't be able to track tool calls properly.

# Test Package
# Compact Eval pack: An eval pack is a small set of test cases used to test and validate the agent behaviour.

eval_cases = [
    {
        "name": "refund_policy",
        "input": "Can I get a refund after joining the course?",
        "expected_tool": "course_policy_search",
        "expected_keywords": ["7 days", "refund", "30 days"],
        "failure_type_if_bad": "weak retrieval or missing retrieval"
    },
    {
        "name": "ticket_status",
        "input": "Check status of ticket TKT-1002",
        "expected_tool": "get_ticket_status",
        "expected_keywords": ["approved", "batch"],
        "failure_type_if_bad": "wrong tool"
    },
    {
        "name": "project_extension_policy",
        "input": "What happens if I miss my backend project deadline?",
        "expected_tool": "course_policy_search",
        "expected_keywords": ["extension", "7 days", "deadline"],
        "failure_type_if_bad": "weak retrieval"
    },
    {
        "name": "policy_plus_ticket",
        "input": "My ticket is TKT-1003. Why was my project extension rejected?",
        "expected_tool": "both",
        "expected_keywords": ["rejected", "extension", "already used"],
        "failure_type_if_bad": "wrong tool or incomplete tool usage"
    },
    {
        "name": "out_of_domain",
        "input": "Who won yesterday's cricket match?",
        "expected_tool": "none",
        "expected_keywords": ["course", "support", "policy", "ticket"],
        "failure_type_if_bad": "over-answering or hallucination"
    }
]

# Execute the above eval test cases.
for case in eval_cases:
    print("Test Case name: ", case["name"])
    print("Input: ", case["input"])
    print("Expected tool call: ", case["expected_tool"])

    output = ask_agent(case["input"])

    print("Agent Output: ", output)

    # Try to validate expected_keywords
    # Try to validate expected_tool call as well. 


# Agentic RAG
# Some common failures while executing the agentic RAG

# 1. Wrong tool call.
# Fix: Improve tool description. Check prompt again.

# 2. Weak Retrieval
# Embeddings
# chunks size too large/small
# chunk overlap
# K is too low 
# Similarity search is not enough.

# 3. Over Refusal
# too strict prompt / restrictions.
# If we have asked the agent to refuse aggressively 
# Tool description 

# 4. Missing Memory.
# Agent won't be able to retain the context.
# chat_history
# agent_scratchpad
# Not appending chat_history properly.

