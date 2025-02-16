from composio_llamaindex import ComposioToolSet, Action
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.llms import ChatMessage
from llama_index.llms.groq import Groq
import os
import dotenv

dotenv.load_dotenv()

TOKEN_LIMIT = 3000

def truncate_text(text: str, limit: int = TOKEN_LIMIT) -> str:
    """Truncate text to fit within token limit safely."""
    return text[:limit] if len(text) > limit else text

def create_research_agent():
    toolset = ComposioToolSet()
    tools = toolset.get_tools(actions=[
        Action.EXA_SEARCH, Action.GOOGLEDOCS_CREATE_DOCUMENT
    ])

    llm = Groq(model="deepseek-r1-distill-llama-70b", api_key=os.getenv("GROQ_API_KEY"))

    prefix_messages = [
        ChatMessage(
            role="system",
            content="You are a research assistant. Find key information and summarize it clearly."
        )
    ]

    return FunctionCallingAgentWorker(
        tools=tools,
        llm=llm,
        prefix_messages=prefix_messages,
        max_function_calls=2,   
        allow_parallel_tool_calls=False,
        verbose=True,
    ).as_agent()

def chatbot(topic: str, domain: str):
    """Perform research based on the given topic and domain."""
    print("\nConducting research...")

    research_prompt = truncate_text(f"Research the topic '{topic}' in '{domain}' and summarize findings.", TOKEN_LIMIT)

    agent = create_research_agent()
    res = agent.chat(research_prompt)

    response_text = truncate_text(res.response)

    with open("content.txt", "w", encoding="utf-8") as file:
        file.write(response_text)

    print("\nReport saved to 'content.txt'.")

    google_doc_prompt = truncate_text("Save this as a Google Doc and return the link.", TOKEN_LIMIT)
    google_doc_url = agent.chat(google_doc_prompt).response

    print("\nGoogle Doc link: " + google_doc_url)

if __name__ == "__main__":
    topic = input("Enter research topic: ")
    domain = input("Enter domain: ")
    chatbot(topic, domain)
