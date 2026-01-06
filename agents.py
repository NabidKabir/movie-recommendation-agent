from langchain_openai import ChatOpenAI
import os
import asyncio
from dotenv import load_dotenv
from fastmcp import Client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages

#ENV Load

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

