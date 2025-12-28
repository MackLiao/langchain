from operator import itemgetter
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("Initializing components...")
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
llm = ChatOpenAI(model="gpt-5-mini")
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """ Answer the question based only on the following context:
    {context}

    Question: {question}

    Answer:
    """
)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# Without LangChain Expressions Language


def retrieval_chain_without_lcel(query: str):
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.invoke({"context": context, "question": query})
    response = llm.invoke(messages)
    return response.content


# With LCEL


def create_retrieval_chain_with_lcel():
    """Create a retrieval chain with LCEL"""

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    query = "What is the pinecone?"

    # print("=" * 70)
    # result = retrieval_chain_without_lcel(query)
    # print(result)
    # print("=" * 70)

    print("=" * 70)
    retrieval_chain = create_retrieval_chain_with_lcel()
    result = retrieval_chain.invoke({"question": query})
    print(result)
    print("=" * 70)
