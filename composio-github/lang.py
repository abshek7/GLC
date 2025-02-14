from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from  langchain_google_genai import ChatGoogleGenerativeAI
from composio_langchain import ComposioToolSet, Action
from dotenv import load_dotenv
import os

load_dotenv()

required_env_vars = ["GOOGLE_API_KEY", "COMPOSIO_API_KEY"]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=os.getenv("GOOGLE_API_KEY"))
composio_toolset = ComposioToolSet(api_key=os.getenv("COMPOSIO_API_KEY"))


owner = "bharathajjarapu"
repo = "SimpliCheck"
path = "src/App.tsx"

tools = composio_toolset.get_tools(actions=[Action.GITHUB_GET_REPOSITORY_CONTENT])

prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

task = f"Use GITHUB_GET_REPOSITORY_CONTENT to get the contents of {path} from {owner}/{repo}"


try:
    response = agent_executor.invoke({"input": task})
    if isinstance(response, dict) and 'output' in response:
        content = response['output']
        print("Content retrieved successfully")
        with open("contents.txt", "w", encoding="utf-8") as f:
            f.write(content)
        print("File saved as contents.txt")
    else:
        print("Unexpected response format:", response)
        
except Exception as e:
    print(f"Error: {str(e)}")