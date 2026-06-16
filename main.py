import os
import sys

from anthropic import APIStatusError
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import SecretStr
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search_tool(query: str) -> str:
    """
    Tool that searches the web for information

    Args:
        query: The query to search for

    Returns:
        The search result
    """
    print(f"Searching the web for {query}")
    response = tavily.search(query=query, max_results=5)

    parts: list[str] = []
    if answer := response.get("answer"):
        parts.append(f"Summary: {answer}")

    for result in response.get("results", []):
        parts.append(
            "\n".join(
                [
                    f"Title: {result.get('title', 'N/A')}",
                    f"URL: {result.get('url', 'N/A')}",
                    f"Content: {result.get('content', 'N/A')}",
                ]
            )
        )

    return "\n\n".join(parts) if parts else "No search results found."


def create_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY is not set. Add it to langchain-course/.env")
        return ChatAnthropic(
            model_name=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            temperature=0,
            api_key=SecretStr(api_key),
            timeout=None,
            stop=None,
        )

    if provider == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
            temperature=0,
        )

    sys.exit(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        "Set LLM_PROVIDER to 'anthropic' or 'ollama' in .env"
    )


def main():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    print("Hello from langchain-course search agent!")
    print(f"Using LLM provider: {provider}")

    if not os.getenv("TAVILY_API_KEY"):
        sys.exit("TAVILY_API_KEY is not set. Add it to langchain-course/.env")

    llm = create_llm()
    agent = create_agent(model=llm, tools=[search_tool])

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content="Job opening for AI Engineer in Bangalore with langchain experience?")]}
        )
    except APIStatusError as e:
        sys.exit(f"Anthropic API error ({e.status_code}): {e.message}")

    print(result)


if __name__ == "__main__":
    main()
