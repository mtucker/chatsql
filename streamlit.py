import os

import dotenv
import streamlit as st
from langchain import SQLDatabase
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_toolkits.sql.prompt import SQL_PREFIX, SQL_FUNCTIONS_SUFFIX
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.callbacks import StreamlitCallbackHandler, LLMonitorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langchain.schema import SystemMessage, AIMessage

dotenv.load_dotenv()


class MODELS:
    GPT3 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"


llm = ChatOpenAI(temperature=0, model_name=MODELS.GPT4)
db = SQLDatabase.from_uri(os.getenv("DATABASE_URL"))
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

prefix = SQL_PREFIX.format(dialect=toolkit.dialect, top_k=10)
max_token_limit = 2000

messages = [
    SystemMessage(content=prefix),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    AIMessage(content=SQL_FUNCTIONS_SUFFIX),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]
prompt = ChatPromptTemplate(messages=messages)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

msgs = StreamlitChatMessageHistory()
memory = AgentTokenBufferMemory(llm=llm, chat_memory=msgs)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)

monitor_callback = LLMonitorCallbackHandler()

st.set_page_config(
    page_title="ChatSQL",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Chat with a SQL database")

starter_message = "Ask me anything!"
if not msgs.messages or st.sidebar.button("Clear message history"):
    st.chat_message("assistant").write(starter_message)


for msg in msgs.messages:
    if msg.type in ["human", "ai"] and msg.content:
        st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    st_callback = StreamlitCallbackHandler(st.container())
    response = agent_executor(
        {"input": prompt, "history": msgs.messages},
        callbacks=[st_callback, monitor_callback],
        include_run_info=True,
    )
    st.chat_message("assistant").write(response["output"])
