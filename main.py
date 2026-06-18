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
from langchain_tavily import TavilySearch
from pydantic import SecretStr
from tavily import TavilyClient
from typing import List
from pydantic import BaseModel, Field

load_dotenv()


class source(BaseModel):
    """" Schema for the source of the information"""
    title: str = Field(description="The title of the source")
    url: str = Field(description="The URL of the source")
    content: str = Field(description="The content of the source")

class AgentResponse(BaseModel):
    """" Schema for the response of the agent"""
    sources: List[source] = Field(description="The sources of the information")
    answer: str = Field(description="The answer to the question")

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
    tools = [TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))]
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content="Job opening for AI Engineer in Bangalore with langchain experience?")]}
        )
    except APIStatusError as e:
        sys.exit(f"Anthropic API error ({e.status_code}): {e.message}")

    print(result)


if __name__ == "__main__":
    main()
