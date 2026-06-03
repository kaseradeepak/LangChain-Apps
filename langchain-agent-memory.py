# Build a small agent with **memory**
# Which can remember the order id shared by the user in the previous messages.
# Agent can call a order status tracking tool.
# Agent can understand the user query and should be able to call the tools based on the query.

# Compare both with-memory and without-memory scenarios for this agent.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

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
    tools=tool,
    verbose=True
)

chat_history = []

def ask_agent(user_input: str):
    response = agent_executor.invoke(
        {
            "input" : user_input,
            "chat_history" : chat_history
        }
    )

    return response['output']

# Run agent on multiple turns

print("Turn-1")
user_input = "Hi, my order id ORD-102."
print("User Input: ", user_input)
print("AI response: ", ask_agent(user_input))

print("====================================\n")

print("Turn-2")
user_input = "What the status of it ?"
print("User Input: ", user_input)
print("AI response: ", ask_agent(user_input))

print("====================================\n")