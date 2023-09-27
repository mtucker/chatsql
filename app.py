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


st.set_page_config(
    page_title="ChatSQL",
    page_icon="🦜",
    layout="wide",
)

st.title("Chat with a SQL database")


class MODELS:
    GPT_35 = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"


openai_api_key = os.getenv("OPENAI_API_KEY", None)

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key:", os.getenv("OPENAI_API_KEY", ""))
    model = st.selectbox("Model", (MODELS.GPT_35, MODELS.GPT_4))

if not openai_api_key:
    st.warning(
        "Please provide an OpenAI API key. You can get one [here](https://beta.openai.com/)."
    )

else:
    llm = ChatOpenAI(temperature=0, model_name=model, openai_api_key=openai_api_key)
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
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

    starter_message = "Ask me anything about the [Chinook digital media store](https://github.com/lerocha/chinook-database)!"
    if not msgs.messages or st.sidebar.button("Clear message history"):
        st.chat_message("assistant").write(starter_message)

    for msg in msgs.messages:
        if msg.type in ["human", "ai"] and msg.content:
            st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input("How many artists are there?"):
        st.chat_message("user").write(prompt)
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": msgs.messages},
            callbacks=[st_callback, monitor_callback],
            include_run_info=True,
        )
        st.chat_message("assistant").write(response["output"])
