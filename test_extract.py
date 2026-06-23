import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from core_packages.vector_engine import VectorEngine

model_name = "Qwen/Qwen2.5-3B-Instruct"
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.eval().to(device)
    print(f"Model loaded on {device}. Starting extraction...")
    t0 = time.time()
    engine = VectorEngine()
    emotion_vectors = engine.extract_emotion_vectors(
        model, tokenizer, layer_idx=18, k_components=None
    )
    t1 = time.time()
    print(f"Extraction successful! Took {t1-t0:.2f} seconds.")
except Exception as e:
    print("Error:", repr(e))
