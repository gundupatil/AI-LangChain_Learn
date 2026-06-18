import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
TEMPERATURE = 0.0


@tool
def get_product_price(product: str) -> float:
    """lookup the price of a product in the catalog"""
    print(f">>Looking up the price of product='{product}'")
    prices = {"laptop": 1000, "mouse": 10, "keyboard": 20}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    apply a discount tier to the price of a product and return the discounted price
    available discount tiers are: "gold", "silver", "bronze"
    """
    print(
        f">> executing apply_discount with price='{price}' "
        f"and discount_tier='{discount_tier}'"
    )
    discount_percentages = {
        "gold": 10,
        "silver": 5,
        "bronze": 15,
    }
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=TEMPERATURE)
    llm_with_tools = llm.bind_tools(tools)

    print(f">> Question:'{question}'")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that can lookup prices and apply "
                "discounts to products.\n"
                "STRICT_RULES - You must follow these exactly:\n"
                "1. Never guess or assume any product price.\n"
                "2. You must call get_product_price to lookup the price of a product.\n"
                "3. Only call apply_discount after you have looked up the price."
            )
        ),
        HumanMessage(content=question),
    ]

    for i in range(MAX_ITERATIONS):
        print(f">> Iteration {i + 1} of {MAX_ITERATIONS}")

        response = llm_with_tools.invoke(messages)
        tool_calls = response.tool_calls
        print(f">> tool_calls: {tool_calls}")

        if not tool_calls:
            print(f">> Final Answer: {response.content}")
            return response.content

        messages.append(response)

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f">> Executing tool: {tool_name}({tool_args})")

            result = tools_dict[tool_name].invoke(tool_args)
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    print(">> Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print(">> Hello, LangChain Agent! (.bind_tools)")
    print()
    run_agent("What is the price of a laptop after applying a gold discount?")
