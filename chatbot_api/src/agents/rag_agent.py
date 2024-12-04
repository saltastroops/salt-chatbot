import dotenv
import os

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models.base import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool

from models.response import Response
from retrievers.document_retriever import retriever
from utils.response_parser import parse

dotenv.load_dotenv()

llm = ChatOpenAI(model=os.getenv("SALT_CfP_MODEL"), temperature=0)

retriever_tool = create_retriever_tool(
    retriever,
    "call-for-proposals-retriever",
    "Query a retriever to get information about the SALT Call for Proposals",
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind_functions([retriever_tool, Response])

agent = (
        {
            "input": lambda x: x["input"],
            # Format agent scratchpad from intermediate steps
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | parse
)

salt_cfp_rag_agent_executor = AgentExecutor(tools=[retriever_tool], agent=agent, verbose=True)
