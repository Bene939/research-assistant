from fastapi import FastAPI
from language_model import load_conversation_chain
import uvicorn

app = FastAPI()
assistant = load_conversation_chain()

@app.get("/assistant_response")
def assistant_response(question: str):
    print("received question: ", question)
    return assistant.predict(input=question)

uvicorn.run(app, host="127.0.0.1", port=8000)