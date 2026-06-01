from langchain-agent-executor import agent_executor

TEST_PACK = [
    {
        "id": "T1_SINGLE_TOOL_STATUS",
        "query": "What is the status of order ORD-101?",
        "query_class": "single-tool",
        "expected_tools": ["get_order_status"]
    },
    {
        "id": "T2_SINGLE_TOOL_DELIVERY",
        "query": "When will order ORD-101 arrive?",
        "query_class": "single-tool",
        "expected_tools": ["estimate_delivery_time"]
    },
    {
        "id": "T3_MULTI_TOOL_STATUS_REFUND",
        "query": "For order ORD-102, check the status and also tell me the refund amount.",
        "query_class": "multi-tool",
        "expected_tools": ["get_order_status", "calculate_refund"]
    },
    {
        "id": "T4_NO_TOOL_CONCEPTUAL",
        "query": "Explain what refund means in simple words.",
        "query_class": "no-tool",
        "expected_tools": []
    },
    {
        "id": "T5_MISSING_ORDER_ID",
        "query": "Can you check my order status?",
        "query_class": "missing-information",
        "expected_tools": []
    }
]

def run_test_pack():
    for test_case in TEST_PACK:
        print("Test Case Id: ", test_case["id"])
        print("Test Case class: ", test_case["query_class"])
        print("Test Case query: ", test_case["query"])

        result = agent_executor.invoke(
            {
               "input" :  test_case["query"]
            }
        )

        # Compare the list of expected tools and actual tools.
