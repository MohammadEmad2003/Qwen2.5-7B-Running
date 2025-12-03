###########################################
#   FASTAPI + NGROK + QWEN2.5 7B 4-BIT    #
###########################################

import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok
import nest_asyncio
import uvicorn
import threading
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

###########################################
# 1. NGROK SETUP
###########################################

nest_asyncio.apply()
ngrok.set_auth_token("2sqtdHtkVxSB7WlK2HOSlJz8ETz_J7RgSHsHSVPwmU1hwpoR")

reserved_domain = "burdened-karyn-empirically.ngrok-free.app"
public_url = ngrok.connect(addr=8000, hostname=reserved_domain)

print("Public URL:", public_url)

###########################################
# 2. LOAD QWEN MODEL IN 4-BIT (NO OOM)
###########################################

print("Loading Qwen2.5-7B in 4-bit...")

model_name = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

print("Model loaded in 4-bit successfully! VRAM usage is now safe.")

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
    # Empty prompt guard
    if not request.headers.strip():
        return {"response": '{"action":"done","reason":"empty input"}'}

    print("[Received Input LEN]:", len(request.headers))

    # Chat template
    messages = [
        {"role": "system", "content": "You are Qwen, a JSON-only web automation agent."},
        {"role": "user", "content": request.headers}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize with truncation (Qwen allows 1M context but Kaggle GPU cannot)
    model_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192,      # safe for 7B on 4-bit
    ).to(model.device)

    # Generate minimal JSON response (VRAM safe)
    with torch.inference_mode():
        output = model.generate(
            **model_inputs,
            max_new_tokens=64,     # agent only needs 1 JSON object
            do_sample=False,
            temperature=0.0
        )

    output_ids = output[0][len(model_inputs.input_ids[0]):]
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
