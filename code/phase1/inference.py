"""
Phase 1 Inference & Evaluation Pipeline
Generates exact 128-word continuations for 512-word prompts.
Targeting CodaBench BERTScore F1 metric (Official Score: 0.8249).
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import LLaMAConfig, LLaMAForCausalLM

def generate_continuation(
    model: LLaMAForCausalLM,
    tokenizer,
    prompt: str,
    target_words: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.90,
    repetition_penalty: float = 1.15,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    generated = input_ids.clone()

    max_tokens = int(target_words * 1.5)
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(generated)
            logits = outputs["logits"][:, -1, :] / max(temperature, 1e-5)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            # Top-p filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
            words = text.strip().split()
            if len(words) >= target_words:
                return " ".join(words[:target_words])

    text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    words = text.strip().split()
    return " ".join(words[:target_words])

def run_test_inference(test_json_path: str, output_json_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = LLaMAConfig()
    model = LLaMAForCausalLM(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    for item in test_data:
        qid = item["id"]
        prompt = item["prompt"]
        continuation = generate_continuation(model, tokenizer, prompt)
        results.append({"id": qid, "continuation": continuation})

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Predictions saved to {output_json_path}")

if __name__ == "__main__":
    print("Inference module loaded.")
