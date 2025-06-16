import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

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
      Wrap the output in this formart and provide no other test\n{fomart_instructions}
      """,
    ),
    ("placeholder", "{chat history}"),
    ("human","{query} {name}"),
    ("placeholder","{agent_scratchpad}"),
  ]
).partial(fomart_instructions=parser.get_format_instructions())


#function to create a simple agent
agent = create_tool_calling_agent(
  llm=llm,
  prompt=prompt,
  tools=[]
)

#Create an executor
agent_executor = AgentExecutor(agent= agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query": "What is the capital of France?", "name":"Alice"})
print(raw_response)

#Create a structured response 
structured_response = parser.parse(raw_response.get("output")[0]["text"])
print(structured_response)



