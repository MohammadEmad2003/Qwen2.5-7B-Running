# qwen_model_server.py
"""
FastAPI + Qwen2.5 model server (defensive).
- Use device_map="auto" + low_cpu_mem_usage to reduce peak memory.
- Support optional offload folder (MODEL_OFFLOAD_FOLDER).
- Default to MAX_NEW_TOKENS=64 (change via env var).
- Compress incoming prompts to reduce memory/useful when callers send full page HTML.
- Always return JSON; on exceptions return {"error":..., "traceback":...}.
- Prints raw responses and errors for easier debugging.
"""
import os
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
import threading
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# NGROK imports (if you use ngrok)
try:
    from pyngrok import ngrok
    import nest_asyncio
    nest_asyncio.apply()
    NGROK_AVAILABLE = True
except Exception:
    NGROK_AVAILABLE = False

# Config via environment variables
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", None)  # if you want a reserved domain
MODEL_OFFLOAD_FOLDER = os.environ.get("MODEL_OFFLOAD_FOLDER", None)
MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "64"))
MAX_HTML_CHARS = int(os.environ.get("QWEN_MAX_HTML_CHARS", "8000"))
TORCH_DTYPE = os.environ.get("QWEN_TORCH_DTYPE", "bfloat16")  # choose 'float16' if bfloat16 unsupported
PYTORCH_CUDA_ALLOC_CONF = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF

app = FastAPI()

# Optional ngrok public URL
public_url = None
if NGROK_AVAILABLE and NGROK_DOMAIN:
    try:
        ngrok.set_auth_token(os.environ.get("NGROK_AUTH_TOKEN", None))
        public_url = ngrok.connect(addr=8000, hostname=NGROK_DOMAIN)
        print("Public URL (ngrok):", public_url)
    except Exception:
        print("ngrok setup failed; continuing without ngrok.")

# Pydantic request model
class PredictRequest(BaseModel):
    headers: str


def _compress_html(html: str, max_chars: int = MAX_HTML_CHARS) -> str:
    """Same heuristic used in controllers to keep context concise by default."""
    if not html:
        return ""
    import re
    try:
        title_m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else ""
    except Exception:
        title = ""
    headings = re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", html, flags=re.I | re.S)[:5]
    headings_text = " | ".join([re.sub(r"<[^>]+>", "", h).strip() for h in headings if h])
    inputs = re.findall(r"<(input|textarea|select)[^>]*>", html, flags=re.I | re.S)[:20]
    inputs_text = []
    for t in inputs:
        s = t
        name = re.search(r'name=["\']([^"\']+)["\']', s)
        ph = re.search(r'placeholder=["\']([^"\']+)["\']', s)
        aria = re.search(r'aria-label=["\']([^"\']+)["\']', s)
        parts = []
        if name:
            parts.append(f"name={name.group(1)}")
        if ph:
            parts.append(f"placeholder={ph.group(1)}")
        if aria:
            parts.append(f"aria={aria.group(1)}")
        if parts:
            inputs_text.append("|".join(parts))
    inputs_summary = "; ".join(inputs_text)
    head = re.sub(r"<[^>]+>", "", html[:max_chars])
    pieces = []
    if title:
        pieces.append(f"TITLE: {title}")
    if headings_text:
        pieces.append(f"HEADINGS: {headings_text}")
    if inputs_summary:
        pieces.append(f"INPUTS: {inputs_summary}")
    pieces.append(f"SNIPPET: {head}")
    result = "\n\n".join(pieces)
    if len(result) > max_chars:
        return result[:max_chars]
    return result


# --- Load tokenizer + model defensively
print("Loading tokenizer & model:", MODEL_NAME)
tokenizer = None
model = None

torch_dtype = None
if TORCH_DTYPE.lower() == "bfloat16" and hasattr(torch, "bfloat16"):
    torch_dtype = torch.bfloat16
elif TORCH_DTYPE.lower() == "float16" and hasattr(torch, "float16"):
    torch_dtype = torch.float16
else:
    torch_dtype = None  # let transformers infer

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
except Exception as e:
    print("Tokenizer load error:", e)
    traceback.print_exc()
    tokenizer = None

try:
    load_kwargs = dict(
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    if MODEL_OFFLOAD_FOLDER:
        os.makedirs(MODEL_OFFLOAD_FOLDER, exist_ok=True)
        load_kwargs["offload_folder"] = MODEL_OFFLOAD_FOLDER

    # Attempt to load model with given device_map. This reduces peak memory and allows offload.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    model.eval()
    print("Model loaded (device_map=auto) successfully.")
except Exception as e:
    print("Model load failed with device_map=auto:", e)
    traceback.print_exc()
    print("Falling back to CPU-only load to avoid OOM.")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": "cpu"})
        model.eval()
        print("Model loaded to CPU.")
    except Exception as e2:
        print("CPU fallback load failed:", e2)
        traceback.print_exc()
        model = None


@app.get("/")
def root():
    return {"message": "Qwen2.5 FastAPI Server Running!", "ngrok_url": str(public_url) if public_url else None}


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Expects JSON: {"headers": "<prompt_or_full_html>"}
    Returns JSON: {"response": "<model text>"} or {"error": "...", "traceback": "..."}
    """
    prompt_raw = request.headers
    print("[Received Input length]:", len(prompt_raw) if prompt_raw else 0)

    try:
        if model is None or tokenizer is None:
            return {"error": "model_not_loaded", "detail": "Model or tokenizer not available on server."}

        # Compress large HTML automatically to reduce memory unless caller explicitly sets a special header string.
        # If you truly want the full raw prompt, set the incoming JSON to include a key like {"headers": "FULL::" + html}
        prompt_for_model = prompt_raw
        if prompt_raw.startswith("FULL::"):
            # caller explicitly asked for full context; strip prefix
            prompt_for_model = prompt_raw[len("FULL::"):]
        else:
            prompt_for_model = _compress_html(prompt_raw, max_chars=MAX_HTML_CHARS)

        # Prepare chat-like input if tokenizer supports it (matching your previous usage)
        messages = [
            {"role": "system", "content": "You are Qwen, a helpful AI assistant."},
            {"role": "user", "content": prompt_for_model}
        ]
        # Note: tokenizer.apply_chat_template is used in your prior code; keep it if available.
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # fallback: simple join
            text = "\n\n".join([m["content"] for m in messages])

        # Tokenize: keep on CPU first
        with torch.inference_mode():
            # Free caches before tokenization/generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model_inputs = tokenizer([text], return_tensors="pt", truncation=False)
            # Attempt to move inputs to model device (works if model is single-device; with sharded models, embedding might be elsewhere)
            try:
                device_of_param = next(model.parameters()).device
                model_inputs = {k: v.to(device_of_param) for k, v in model_inputs.items()}
            except Exception:
                # if moving fails, proceed and let model handle placement (device_map=auto handles inputs usually)
                pass

            # Constrain max_new_tokens to a sane upper bound to avoid big allocations
            max_new = min(MAX_NEW_TOKENS, 512)

            # Generate under autocast if using CUDA and bfloat16/float16
            try:
                if torch.cuda.is_available() and (torch_dtype is not None):
                    with torch.cuda.amp.autocast():
                        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new)
                else:
                    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new)
            except torch.cuda.OutOfMemoryError as oom:
                # On OOM, clear caches and return JSON error (do not raise HTML traceback)
                print("CUDA OOM during generation:", oom)
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {"error": "CUDA_OOM", "detail": str(oom)}
            except Exception as e:
                print("Generation error:", e)
                traceback.print_exc()
                return {"error": "generation_error", "detail": str(e)}

            # decode
            try:
                # take only the newly generated tokens
                if isinstance(generated_ids, tuple) or isinstance(generated_ids, list):
                    # some libs return tuple; handle first element
                    gen = generated_ids[0]
                else:
                    gen = generated_ids
                # compute slice start
                input_len = model_inputs["input_ids"].shape[1] if "input_ids" in model_inputs else 0
                output_ids = gen[0][input_len:]
                response_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            except Exception as e:
                print("Decode error:", e)
                traceback.print_exc()
                return {"error": "decode_error", "detail": str(e)}

            # Always return valid JSON
            print("Generation successful (len=%d). Returning JSON response." % (len(response_text)))
            return {"response": response_text}

    except Exception as e:
        tb = traceback.format_exc()
        print("Predict endpoint exception:", tb)
        return {"error": "server_exception", "detail": str(e), "traceback": tb}


# run uvicorn when executed directly (use environment to control reload)
if __name__ == "__main__":
    import uvicorn
    print("Starting Qwen model server with uvicorn...")
    uvicorn.run("qwen_model_server:app", host="0.0.0.0", port=8000, log_level="info")
