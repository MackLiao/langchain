from dotenv import load_dotenv
load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(temperature=0, model="gpt-5-mini")
tools = [TavilySearch()]
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor

def main():
    result = chain.invoke({"messages":HumanMessage(content="Search for 3 job posting for an AI engineer in the United States in LinkedIn")})
    print(result)

if __name__ == "__main__": 
    main()