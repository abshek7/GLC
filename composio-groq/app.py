from composio_llamaindex import App, ComposioToolSet
from llama_index.core.agent import FunctionCallingAgentWorker 
from llama_index.core.llms import ChatMessage
from llama_index.llms.groq import Groq 
import os
import dotenv

dotenv.load_dotenv()


toolset = ComposioToolSet() 
tools = toolset.get_tools(apps=[App.TAVILY]) 


llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))


prefix_messages = [
    ChatMessage(
        role="system", 
        content="You are now a Search analyst agent, and whatever you are requested, you will try to execute utilizing your tools."
    )
]

agent = FunctionCallingAgentWorker(
    tools=tools, 
    llm=llm, 
    prefix_messages=prefix_messages, 
    max_function_calls=10, 
    allow_parallel_tool_calls=False, 
    verbose=True
).as_agent()

human_input = "write a summary of 200 words on what is the current update on bharathajjarapu from his GitHub profile."
response = agent.chat("Task to perform: " + human_input)


response_text = response.response  
with open("content.txt", "w", encoding="utf-8") as file:
    file.write(response_text)

print("Response saved to content.txt")
