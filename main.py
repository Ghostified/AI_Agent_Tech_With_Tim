import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
#deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_api_key = os.getenv("OPEN_ROUTER_API_KEY")

load_dotenv()

#llm = ChatOpenAI(model="gpt-4o-mini")
#llm=ChatAnthropic(model="claude-opus-4-20250514")
llm = ChatOpenAI(model="gpt-4o-mini",
                 openai_api_base ="https://openrouter.ai/api/v1",
                 openai_api_key =deepseek_api_key
                 )

response = llm.invoke("What is the meaning of life?")
print(response)


