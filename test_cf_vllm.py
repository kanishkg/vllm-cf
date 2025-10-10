import vllm
import torch
from transformers import AutoTokenizer
import Levenshtein

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

llm = vllm.LLM(model=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

SYSTEM_PROMPT = "Be creative and keep your response as short as possible."
USER_PROMPT = "Tell me a fantasy story about a captain. The story should have either a happy or a sad ending."
SPLIT_AT = ""
sampling_params = vllm.SamplingParams(temperature=0.1, seed=0, max_tokens=512)
message = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
]
prompt = tokenizer.apply_chat_template(message, tokenize=False)
outputs_factual = llm.generate([prompt], sampling_params)
output_factual_text = outputs_factual[0].outputs[0].text

print("========== Factual ==========")
print(output_factual_text)
SPLIT_AT = "Lyra"
REPLACE_WITH = "Maeve"
prefix = output_factual_text.split(SPLIT_AT)[0] + REPLACE_WITH

# counterfactual
message_prefix = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
    {"role": "assistant", "content": prefix},
]
prompt_prefix = tokenizer.apply_chat_template(message_prefix, tokenize=False)
prompt_prefix = prompt_prefix.replace("<|eot_id|>", "")
sampling_params = vllm.SamplingParams(temperature=0.1, seed=0, max_tokens=512)
outputs_cf = llm.generate([prompt_prefix], sampling_params)
output_cf_text = prefix + outputs_cf[0].outputs[0].text

print("========== Counterfactual ==========")
print(output_cf_text)

# interventional
sampling_params = vllm.SamplingParams(temperature=0.1, seed=1, max_tokens=512)
outputs_interventional = llm.generate([prompt_prefix], sampling_params)
output_interventional_text = prefix + outputs_interventional[0].outputs[0].text
print("========== Interventional ==========")
print(output_interventional_text)


# get edit distance between factual and counterfactual
edit_distance = Levenshtein.distance(output_factual_text, output_cf_text)
print(f"Edit distance between factual and counterfactual: {edit_distance}")
edit_distance = Levenshtein.distance(output_factual_text, output_interventional_text)
print(f"Edit distance between factual and interventional: {edit_distance}")