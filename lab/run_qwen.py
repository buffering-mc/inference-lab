from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda")

msgs = [{"role": "user", "content": "Explain the KV cache in one paragraph."}]
inputs = tok.apply_chat_template(
    msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("cuda")
out = model.generate(**inputs, max_new_tokens=100)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")