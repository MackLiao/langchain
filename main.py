from typing import List

from pydantic import BaseModel, Field
from pyexpat import model
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response"""
    answer:str = Field(description="The agent's response to the query")
    sources:List[Source] = Field(default_factory=list, description="The sources used to answer the query")

llm = ChatOpenAI(temperature=0, model="gpt-5-mini")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job posting for an AI engineer in the United States in LinkedIn")})
    print(result)


if __name__ == "__main__":
    main()
