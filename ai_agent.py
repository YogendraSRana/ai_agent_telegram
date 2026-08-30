import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

from tools import (
    get_current_datetime,
    get_news,
    calculate,
    list_files,
    send_file,
)

load_dotenv()
gemini_key = os.getenv("GOOGLE_API_KEY")

gemini = init_chat_model(
    "google_genai:gemini-3.1-flash-lite",
    api_key=gemini_key
)

tools = [
    get_current_datetime,
    get_news,
    calculate,
    list_files,
    send_file,
]

system_prompt = """You are a fast, intelligent AI Assistant on Telegram.
- For current date, day, month, year, or time: always call `get_current_datetime`.
- For any live news or current events: call `get_news`.
- For math calculations: call `calculate`.
- For sending files: call `send_file`.
Respond concisely and immediately."""

agent = create_agent(
    model=gemini,
    tools=tools,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(model=gemini, trigger=("messages", 20), keep=("messages", 6))
    ],
    system_prompt=system_prompt
)