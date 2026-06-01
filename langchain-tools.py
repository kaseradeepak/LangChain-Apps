import json
from typing import Literal
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import ToolMessage

MODEL_NAME="gpt-5.2"

# Fake Orders Database.
ORDERS = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "item": "Wireless Mouse",
        "status": "shipped",
        "eta": "Tomorrow",
        "payment_status": "paid"
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "item": "Mechanical Keyboard",
        "status": "delivered",
        "eta": "Delivered yesterday",
        "payment_status": "paid"
    },
    "ORD-1003": {
        "order_id": "ORD-1003",
        "item": "USB-C Cable",
        "status": "not_shipped",
        "eta": "Waiting for payment confirmation",
        "payment_status": "pending"
    }
}

REFUND_TICKETS = {}

POLICIES = {
    "refund": (
        "Refunds are allowed within 7 days of delivery for damaged, wrong, "
        "or defective items. Refunds are processed after inspection."
    ),
    "shipping": (
        "Standard shipping takes 3-5 business days. Express shipping takes "
        "1-2 business days."
    ),
    "warranty": (
        "Electronics have a 1-year manufacturer warranty. Physical damage is "
        "not covered."
    )
}

# Tool-1
class OrderStatusInput(BaseModel):
    order_id: str = Field(description="Order id should be in the format of ORD-1001")

@tool(args_schema=OrderStatusInput)
def get_order_status(order_id: str):
    """
    Fetch order status, item details, payment status, and delivery ETA.

    Use this tool only when the user asks about the status, delivery, shipping, or ETA of a specific order id.
    """
    order_id = order_id.upper()
    order = ORDERS.get(order_id)

    if not order:
        return json.dumps({
            "ok": False,
            "error_type": "ORDER_NOT_FOUND",
            "message": f"No order found for order_id={order_id}"
        })

    return json.dumps({
        "ok": True,
        "order": order
    })

# Tool-2
class PolicyLookupInput(BaseModel):
    topic: Literal["refund", "shipping", "warranty"] = Field(
        description="Policy topic. Must be one of: refund, shipping, warranty."
    )

@tool(args_schema=PolicyLookupInput)
def lookup_policy(topic: str) -> str:
    """
    Look up company policy for refund, shipping, or warranty.

    Use this tool when the user asks about rules, eligibility, timelines, or company policy.
    """
    return json.dumps({
        "ok": True,
        "topic": topic,
        "policy": POLICIES[topic]
    })

# Tool 3: Create refund ticket
class RefundTicketInput(BaseModel):
    order_id: str = Field(
        description="Order id in format ORD-1001, ORD-1002, etc."
    )

    reason: Literal["damaged", "late_delivery", "wrong_item", "other"] = Field(
        description="Reason for refund request."
    )

    customer_note: str = Field(
        description="Short customer explanation for the refund request."
    )

@tool(args_schema=RefundTicketInput)
def create_refund_ticket(
    order_id: str,
    reason: str,
    customer_note: str
) -> str:
    """
    Create a refund ticket for an existing paid order.

    Use this tool only when the user explicitly wants to raise, create, open, or file a refund request.
    """
    order_id = order_id.upper()
    order = ORDERS.get(order_id)

    if not order:
        return json.dumps({
            "ok": False,
            "error_type": "ORDER_NOT_FOUND",
            "message": f"Cannot create refund. No order found for {order_id}."
        })

    if order["payment_status"] != "paid":
        return json.dumps({
            "ok": False,
            "error_type": "PAYMENT_NOT_COMPLETED",
            "message": (
                f"Cannot create refund for {order_id} because payment status "
                f"is {order['payment_status']}."
            )
        })

    ticket_id = f"RF-{len(REFUND_TICKETS) + 1:04d}"

    REFUND_TICKETS[ticket_id] = {
        "ticket_id": ticket_id,
        "order_id": order_id,
        "reason": reason,
        "customer_note": customer_note,
        "status": "created"
    }

    return json.dumps({
        "ok": True,
        "ticket_id": ticket_id,
        "message": (
            f"Refund ticket {ticket_id} created successfully for {order_id}."
        )
    })

# Register tools
tools = [
    get_order_status,
    lookup_policy,
    create_refund_ticket
]

tools_by_name = {}
for tool in tools:
    tools_by_name[tool.name] = tool

model = init_chat_model(MODEL_NAME, temperature=1)

model_with_tools = model.bind_tools(tools)

# Safe tool execution
def execute_tool_call_safely(tool_call: dict) -> ToolMessage:
    """
    Execute a model-emitted tool call safely.

    Instead of crashing the program, convert tool errors into ToolMessage
    objects so the model can recover and respond gracefully.
    """
    tool_name = tool_call.get("name")
    tool_call_id = tool_call.get("id")

    selected_tool = tools_by_name.get(tool_name)

    if selected_tool is None:
        return ToolMessage(
            content=json.dumps({
                "ok": False,
                "error_type": "UNKNOWN_TOOL",
                "message": f"Tool '{tool_name}' is not available."
            }),
            tool_call_id=tool_call_id
        )

    try:
        return selected_tool.invoke(tool_call)

    except Exception as error:
        return ToolMessage(
            content=json.dumps({
                "ok": False,
                "error_type": error.__class__.__name__,
                "message": str(error)
            }),
            tool_call_id=tool_call_id
        )

# Manual tool-feedback loop
def run_customer_support_agent(user_query: str, max_steps: int = 5) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful customer support assistant. "
                "Use tools when you need order data, policy data, or when "
                "the user wants to create a refund ticket."
                "If a tool returns an error, explain the issue clearly and "
                "ask the user for the missing or corrected information."
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    for step in range(max_steps):
        ai_msg = model_with_tools.invoke(messages)
        messages.append(ai_msg)

        print(f"\n--- Step {step + 1}: Model response ---")
        print("Content:", ai_msg.content)
        print("Tool calls:", ai_msg.tool_calls)

        if not ai_msg.tool_calls:
            return ai_msg.content

        for tool_call in ai_msg.tool_calls:
            tool_msg = execute_tool_call_safely(tool_call)
            messages.append(tool_msg)

            print("\n--- Tool result ---")
            print("Tool call id:", tool_msg.tool_call_id)
            print("Content:", tool_msg.content)

    return "I could not complete the request within the allowed number of steps."

queries = [
        "Create a refund ticket for ORD-9999 because the item is damaged."
    ]

for query in queries:
    print("\n" + "=" * 80)
    print("USER:", query)
    final_answer = run_customer_support_agent(query)
    print("\nFINAL ANSWER:")
    print(final_answer)

# "Where is my order ORD-1001?"
# "Can I get a refund if my item is damaged?",
#         "Create a refund ticket for ORD-1002 because the item is damaged.",
#         "Create a refund ticket for ORD-9999 because the item is damaged.",
#         "Tell me a joke about databases."




