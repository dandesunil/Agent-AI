from fastapi import FastAPI
from models.model import Chat
from services.llm import MistralLangGraphAgent
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder

app = FastAPI()

@app.get("/")
async def base():
    return {"message": "Agent API is running."}

@app.post("/chat")
async def chat(request_data: Chat):
    payload = jsonable_encoder(request_data)
    q = payload.get("query")
    if not q:
        return JSONResponse({"error": "no query"}, status_code=400) 

    agent = MistralLangGraphAgent()

    answer = agent.run(q)
    return {"answer": answer}

    # query = "What is the square root of 81?"
    # print("🔹 Query:", query)
    # print("🔹 Final Answer:", agent.run(query))

    # query2 = "What’s the capital of France?"
    # print("\n🔹 Query:", query2)
    # print("🔹 Final Answer:", agent.run(query2))



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

