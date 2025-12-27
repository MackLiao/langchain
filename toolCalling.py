from dotenv import load_dotenv

load_dotenv()

from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
# llm = ChatOpenAI(model="gpt-5")

# Bind tools to the LLM for native tool calling
llm_with_tools = llm.bind_tools(tools)
structured_llm = llm.with_structured_output(AgentResponse)

# Create a simple prompt for tool-calling agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the provided tools to answer questions."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(
    llm=llm_with_tools,
    tools=tools,
    prompt=prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor
extract_output = RunnableLambda(lambda x: x["output"])
chain = agent_executor | extract_output | structured_llm


def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
