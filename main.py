import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool

#deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_api_key = os.getenv("OPEN_ROUTER_API_KEY")


load_dotenv()

#Class that defines Prompt Template to create structured responses/schema
class ResearchResponse(BaseModel):
  topic: str
  summary: str
  sources: list[str]
  tools_used: list[str]


#Make API Calls to model

#llm = ChatOpenAI(model="gpt-4o-mini")
#llm=ChatAnthropic(model="claude-opus-4-20250514")
llm = ChatOpenAI(model="gpt-4o-mini",
                 openai_api_base ="https://openrouter.ai/api/v1",
                 openai_api_key =deepseek_api_key
                 )


#Parser to parse the research response
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

#Create a prompt
prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      """ 
      You are a research assistant that will help generate a research paper.
      Answeer the user query and use neccessary tools
      Wrap the output in this formart and provide no other test\n{format_instructions}
      """,
    ),
    ("placeholder", "{chat history}"),
    ("human","{query}"),
    ("placeholder","{agent_scratchpad}"),
  ]
).partial(format_instructions=parser.get_format_instructions())


#function to create a simple agent
tools = [search_tool]
agent = create_tool_calling_agent(
  llm=llm,
  prompt=prompt,
  tools=tools
)

#Create an executor
agent_executor = AgentExecutor(agent= agent, tools=tools, verbose=True)
query = input("What can i help youto research?")
raw_response = agent_executor.invoke({"query": query})


#Create a structured response 
try:
  structured_response = parser.parse(raw_response.get("output")[0]["text"])
  print(structured_response)
except Exception as e:
  print("Error parsing response", e , "Raw Response", raw_response)



