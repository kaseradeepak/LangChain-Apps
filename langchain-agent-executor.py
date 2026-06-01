from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

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
    Get the current status of order id provided in the input.
    Use this tool when the user ask about the order status for a specific order_id.
    """

    order = ORDERS.get(order_id)

    if not order:
        return f"No order found for the given order_id: {order_id}"
    
    return (
        f"Order with id: {order_id} is {order['status']}."
        f"City: {order['city']}."
        f"Order Amount: {order['amount']}"
    )

@tool
def calculate_refund(order_id: str) -> str:
    """
    Calculate the refund amount of the given order_id.
    Use this tool when user asks about refund for a specific order_id. 
    """

    order = ORDERS.get(order_id)

    if not order:
        return f"No order found for the given order_id: {order_id}"

    if order['status'] == 'cancelled':
        return f"Refund amount of the order id: {order_id} is {order['amount']}."

    if order['status'] == 'delivered':
        return (
            f"Order Id: {order_id} has already been delivered. "
            f"Refund eligibility depends on product policy."
        )
    
    return (
        f"Order Id: {order_id} has been shipped."
        f"Refund can't be calculated at this stage."
    )

@tool
def estimate_delivery_time(order_id: str) -> str:
    """
    Estimates delivery timeline for the order_id.
    Use this tool when user asks for the eta or delivery timeline for a specific order_id. 
    """

    order = ORDERS.get(order_id)

    if not order:
        return f"No order found for the given order_id: {order_id}"

    if order['status'] == 'shipped':
        return f"Order {order_id} has been shipped and is expected to arrive in {order['delivery_days']} days."
    
    if order['status'] == 'delivered':
        return f"Order {order_id} has already been delivered"
    
    if order['status'] == 'cancelled':
        return f"Order {order_id} has been cancelled, so there is not delivery timeline estimate."

    return f"Delivery status for order id: {order_id} is not available."

tools = [
    get_order_status,
    calculate_refund,
    estimate_delivery_time
]

llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful e-commerce support assistant.

Rules:
1. Use tools only when required.
2. If the user asks about a specific order, use the relevant tool.
3. If the user asks a general conceptual question, answer directly without tools.
4. If order id is missing, ask the user for the order id.
5. Keep the final answer clear and beginner-friendly.
"""
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]

)

# Agent => prompt + LLM + tools
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Who is going to execute this agent ? => AgentExecutor

