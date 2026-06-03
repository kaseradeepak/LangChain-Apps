# Manual append is okay for learning project but not good for production applications.
# Manual append is prone to errors.

# Langchain provides RunnableWithMessageHistory to automatically: 
# - Load the previous messages.
# - Inject them into the prompt.
# - append the user message and AI message after the response.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Fake Database.
ORDERS = {
    "ORD-101": {
        "status": "shipped",
        "city": "Delhi",
        "amount": 2500,
        "delivery_days": 2
    },
    "ORD-102": {
        "status": "cancelled",
        "city": "Bangalore",
        "amount": 4000,
        "delivery_days": 0
    },
    "ORD-103": {
        "status": "delivered",
        "city": "Mumbai",
        "amount": 1500,
        "delivery_days": 0
    }
}

@tool
def get_order_status(order_id: str) -> str:
    """
    Returns the status of the order id given by the user.

    Use this tool when the user is asking about the order tracking status.
    """

    order = ORDERS.get(order_id)

    if not order:
        return f"Order with the id: {order_id} not found."

    return f"Order status: {order['status']}"

tools = [get_order_status]

# Create llm
llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0
)

#Create a prompt template with messages place holder.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful customer support agent.

Rules:
- If the user gives an order id, remember it for this conversation.
- If the user asks a follow-up like "track it" or "where is it", use the order id from chat history.
- Use tools when order status is required.
- If no order id is available, politely ask the user for the order id.
"""
    ),

    # previous message of the conversation will be inserted in the chat_history.
    # optional=True makes the chat_history parameter as optional.
    MessagesPlaceholder(variable_name="chat_history", optional=True),

    ("human", "{input}"),

    # Tool-Calling agent intermediate steps will be store here.
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Session wise memory store
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Returns the chat history for the given session or conversation.
    """

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    
    return store[session_id]

agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)

def ask_agent(session_id: str, user_input: str):
    response = agent_with_memory.invoke(
        {"input" : user_input},
        config={
            "configurable": {
                "session_id" : session_id
            }
        }
    )

    return response['output']

session_id = "user-001"

print("Turn-1")
user_input = "Hi, My order is ORD-101."
print("User Input: ", user_input)
print("AI Response: ", ask_agent(session_id, user_input))

print("===================================\n")

print("Turn-2")
user_input = "what is the status of it ?"
print("User Input: ", user_input)
print("AI Response: ", ask_agent(session_id, user_input))

# Show the stored memory
print("Stored message in memory:")
for message in store[session_id].messages:
    print(type(message).__name__, " - ", message.content)

# Rolling Conversational History
# To pass the rolling history, we can set the value of  n_messages in MessagesPlaaceholder.
# This keeps only defined number of messages in the history.

# Full History vs Rolling History

# Common Debug points / Errors while building agents with memory using Langchain.
# Wrong placeholder name.
# If we forget to append in the chat_history
# append messages in wrong order.
# Shares common memory across sessions or conversations.