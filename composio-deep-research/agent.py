from composio_llamaindex import ComposioToolSet, Action
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.llms import ChatMessage
from llama_index.llms.groq import Groq
import os
import dotenv

dotenv.load_dotenv()

TOKEN_LIMIT = 5000
QUESTION_LIMIT = 2  

def truncate_text(text: str, limit: int = TOKEN_LIMIT) -> str:
    """Truncate text to fit within token limit safely."""
    return text[:limit] if len(text) > limit else text

def create_research_agent():
    toolset = ComposioToolSet()
    tools = toolset.get_tools(actions=[
        Action.EXA_SEARCH, Action.EXA_SIMILARLINK,
        Action.GOOGLEDOCS_CREATE_DOCUMENT
    ])

    llm = Groq(model="deepseek-r1-distill-llama-70b", api_key=os.environ.get("GROQ_API_KEY"))

    prefix_messages = [
        ChatMessage(
            role="system",
            content=(
                "You are a sophisticated research assistant. Perform comprehensive research on the given query and provide detailed analysis. "
                "Focus on key concepts, developments, stakeholders, relevant data, and implications. "
                "Ensure all information is accurate, up-to-date, and properly sourced."
            ),
        )
    ]

    return FunctionCallingAgentWorker(
        tools=tools,
        llm=llm,
        prefix_messages=prefix_messages,
        max_function_calls=3,
        allow_parallel_tool_calls=False,
        verbose=True,
    ).as_agent()

def generate_questions(topic: str, domain: str) -> list[str]:
    """Generate guiding questions for research, limited to 2."""
    llm = Groq(model="deepseek-r1-distill-llama-70b", api_key=os.environ.get("GROQ_API_KEY"))

    questions_prompt = f"Generate {QUESTION_LIMIT} simple yes/no questions to guide research on '{topic}' in the domain '{domain}'."

    questions_response = llm.complete(truncate_text(questions_prompt, 500))
    
    return [
        q.strip() for q in questions_response.text.strip().split('\n')
        if q.strip() and any(q.startswith(str(i)) for i in range(1, QUESTION_LIMIT + 1))
    ]

def summarize_response(response: str) -> str:
    """Summarize user response if it's too long."""
    if len(response) > 1000:
        summary_prompt = f"Summarize the following response concisely:\n{response}"
        llm = Groq(model="deepseek-r1-distill-llama-70b", api_key=os.environ.get("GROQ_API_KEY"))
        return llm.complete(truncate_text(summary_prompt, 500)).text
    return response

def chatbot():
    print("Hello! I can help you research any topic. Let's start!")

    topic = input("What topic would you like to research: ")
    domain = input("What domain is this topic in: ")

    cleaned_questions = generate_questions(topic, domain)

    print("\nConsider these research questions:")
    print("\n".join(cleaned_questions))

    answer = input("\nProvide responses to these questions: ")

    summarized_answer = summarize_response(answer)

    research_prompt = truncate_text(
        f"Topic: {topic}\nDomain: {domain}\nUser's response:\n{summarized_answer}",
        TOKEN_LIMIT
    )

    print("\nConducting research and creating a report...")

    agent = create_research_agent()
    res = agent.chat(research_prompt)

    response_text = truncate_text(res.response)

    with open("content.txt", "w", encoding="utf-8") as file:
        file.write(response_text)

    print("\nYour research report has been saved to 'content.txt'.")

    google_doc_prompt = truncate_text("Save this content to a new Google Doc and return the document URL: " + response_text)
    google_doc_url = agent.chat(google_doc_prompt).response

    print("\nYour research report is available at: " + google_doc_url)

if __name__ == "__main__":
    chatbot()
