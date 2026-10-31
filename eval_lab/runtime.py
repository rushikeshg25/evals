"""Load the pinned Qwen checkpoint only when inference is requested."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-0.6B"


class LocalModel:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.revision = (ROOT / "model-revision.txt").read_text().strip()
        print("Loading pinned Qwen checkpoint from local cache...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=self.revision, local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=self.revision, local_files_only=True,
            use_safetensors=True, dtype=torch.float32,
        ).to("cpu")
        self.model.eval()

    def generate(self, system_prompt, user_text, max_new_tokens=16):
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        tokens = output[0, inputs["input_ids"].shape[1]:]
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        return {
            "raw_output": self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            "input_tokens": inputs["input_ids"].shape[1],
            "output_tokens": len(tokens),
            "hit_token_limit": len(tokens) >= max_new_tokens and int(tokens[-1]) not in eos_ids,
        }
