"""Compatibility entry point. The implementation lives in eval_lab/runtime.py."""

from eval_lab.runtime import MODEL_ID, ROOT, LocalModel

REVISION = (ROOT / "model-revision.txt").read_text().strip()
_runtime = None


def generate(system_prompt, user_text, max_new_tokens=32):
    global _runtime
    if _runtime is None:
        _runtime = LocalModel()
    return _runtime.generate(system_prompt, user_text, max_new_tokens)["raw_output"]


if __name__ == "__main__":
    print(generate("Answer briefly and clearly.", "What is a database index?", 80))
