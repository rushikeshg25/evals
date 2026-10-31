from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = Path("model-revision.txt").read_text().strip()

print("Loading tokenizer and model...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    local_files_only=True,
    use_safetensors=True,
    dtype=torch.float32,
).to("cpu")

model.eval()

print("Device:", next(model.parameters()).device)
print("Weight dtype:", next(model.parameters()).dtype)

def generate(system_prompt, user_text, max_new_tokens=32):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

if __name__ == "__main__":
    print(generate(
        "Answer briefly and clearly.",
        "What is a database index?",
        max_new_tokens=80,
    ))