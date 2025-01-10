from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import openai
import json

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is missing. Please ensure it is set in the .env file.")
openai.api_key = openai_api_key

# Initialize Chat Model
chat_model = ChatOpenAI(temperature=0.7, model="gpt-4o-mini", openai_api_key=openai_api_key)

# Define Prompt Template
prompt_template = ChatPromptTemplate.from_template("""
You are an intelligent recipe assistant. Respond to user queries as follows:
- Provide detailed, helpful, and user-friendly responses to human food recipe-related questions.
- Structure your response in the following format:
    {{
        "title": "<Recipe Title>",
        "ingredients": ["Ingredient 1", "Ingredient 2", ...],
        "steps": ["Step 1", "Step 2", ...]
    }}
- Ensure the output can be parsed as JSON.
- For unrelated questions, politely inform the user of your scope.

Conversation so far:
{history}

User input: {input}

Your response:
""")

# Set up memory for context
memory = ConversationBufferMemory()

# Create a conversation chain
conversation = ConversationChain(
    llm=chat_model,
    prompt=prompt_template,
    memory=memory
)

# Chatbot function
def recipe_chatbot(user_input: str) -> dict:
    try:
        response = conversation.run(input=user_input)
        # Parse response into a structured format if it's JSON-like
        try:
            structured_response = json.loads(response)
            return structured_response
        except json.JSONDecodeError:
            return {"error": "Response could not be structured.", "text": response}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

# FastAPI app
app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as file:
        return file.read()

class UserInput(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(user_input: UserInput):
    try:
        response = recipe_chatbot(user_input.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
