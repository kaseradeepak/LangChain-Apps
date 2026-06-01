from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

# query = "What is the status of my order: ORD-101 ?"
query = "For order id ORD-102, check the status, tell me the delivery estimate and refund amount, also book a flight for me  from delhi to Bangalore for 5th June."

result = agent_executor.invoke(
    {
        "input" : query
    }
)

print(result['output'])

print("========================================================================")

# print(result["intermediate_steps"])

for index, step in enumerate(result["intermediate_steps"], start=1):
    action, observation = step
    print(f"Step-{index}")
    print(f"Tool selected: {action.tool}")
    print(f"Tool input: {action.tool_input}")
    print(f"Tool observation:", observation)

# return_intermediate_steps
# max_iterations
# verbose -> Shows the internal execution logs.
# handle_parsing_errors -> handle paring erros while calling the tools. It gives a chance to the agent to recover if some issue happens during parsing. 

# MessagesPlaceholder(variable_name="agent_scratchpad")
# The agentscratchpad stores the tool calls and tool obervations.
# agent_scractchpad is the agent's working memory during one request. It stores what tool was called and what result we received.
# Without agent_scractchpad, agent cannot properly continue with multiple tools execution.

# Step-Level Observability

# Cohort Test Pack => Test Cases
# A test pack is a collection of predefined queries to validate the agent behaviours.

# For Agentic AI application, checking only input and putput is not enough.
# Rather we need to check the complete journey (intermediate steps)
