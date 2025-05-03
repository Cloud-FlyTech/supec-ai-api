from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

model_id = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.get("/")
def read_root():
    return {"message": "Supec AI is live!"}

@app.post("/chat")
def generate_response(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    output = pipe(messages, max_new_tokens=100)
    return {"response": output[0]["generated_text"]}