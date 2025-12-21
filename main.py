from pyexpat import model
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


llm = ChatOpenAI(temperature=0, model="gpt-5-mini")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke({"messages":HumanMessage(content="what's the weather in Hangzhou?")})
    print(result)


if __name__ == "__main__":
    main()
