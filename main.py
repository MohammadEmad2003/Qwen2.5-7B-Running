###########################################
#   FASTAPI + NGROK + QWEN2.5 7B BACKEND  #
###########################################

from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok
import nest_asyncio
import uvicorn
import threading
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

###########################################
# 1. NGROK SETUP
###########################################

nest_asyncio.apply()
ngrok.set_auth_token("2sqtdHtkVxSB7WlK2HOSlJz8ETz_J7RgSHsHSVPwmU1hwpoR")

reserved_domain = "burdened-karyn-empirically.ngrok-free.app"
public_url = ngrok.connect(addr=8000, hostname=reserved_domain)

print("Public URL:", public_url)

###########################################
# 2. LOAD QWEN MODEL
###########################################

print("Loading model...")

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded successfully!")

###########################################
# 3. FASTAPI SETUP
###########################################

class PredictRequest(BaseModel):
    headers: str

app = FastAPI()


@app.get("/")
def root():
    return {
        "message": "Qwen2.5 FastAPI Server Running!",
        "ngrok_url": str(public_url)
    }


@app.post("/predict")
def predict(request: PredictRequest):
    print("[Received Input]:", request.headers)

    # Prepare prompt for Qwen
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful AI assistant."},
        {"role": "user", "content": request.headers}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    response_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    result = {"response": response_text}
    print(result)

    return result


###########################################
# 4. START FASTAPI IN A BACKGROUND THREAD
###########################################

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

###########################################
# 5. KEEP THE APP RUNNING FOREVER
###########################################

while True:
    time.sleep(1)
