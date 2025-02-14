
from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel

# Define chatbot agents
agents = {
    "tech_support": ChatOpenAI(model_name="gpt-4"),
    "sales": ChatOpenAI(model_name="gpt-4"),
    "general": ChatOpenAI(model_name="gpt-4")
}

app = FastAPI()

class ChatRequest(BaseModel):
    agent: str
    message: str

@app.post("/chat/")
def chat(request: ChatRequest):
    agent = agents.get(request.agent, agents["general"])
    response = agent.predict(request.message)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
