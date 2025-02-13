from composio_gemini import Action, ComposioToolSet, App
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json

load_dotenv()

required_env_vars = ["GOOGLE_API_KEY", "COMPOSIO_API_KEY"]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

# Create composio client,toolset
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
toolset = ComposioToolSet(api_key=os.getenv("COMPOSIO_API_KEY"))



target="harshbhaskerwar28"
target_repo = "harshbhaskerwar28/Care-AI"


#genai client config
config = types.GenerateContentConfig(
    tools=toolset.get_tools( 
        actions=[
            Action.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER,
            Action.GITHUB_SEARCH_TOPICS
        ]
    )
)

chat = client.chats.create(model="gemini-2.0-flash", config=config)

response = chat.send_message(
   f"Can you star `{target_repo}` repository on github",
   f"Can you search for repositories with topic `python`"

)



print(response.text)
