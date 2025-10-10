import vllm
import torch

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

SYSTEM_PROMPT = "Be creative and keep your response as short as possible."
USER_PROMPT = "Tell me a fantasy story about a captain. The story should have either a happy or a sad ending."
SPLIT_AT = ""


llm = vllm.LLM(model=MODEL_NAME)
sampling_params = vllm.SamplingParams(temperature=0.1, seed=0)
message = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
]
outputs_factual = llm.generate([message], sampling_params)
output_factual_text = outputs_factual[0].outputs[0].text

print("========== Factual ==========")
print(output_factual_text)

prefix = output_factual_text.split(SPLIT_AT)[0] + SPLIT_AT

# counterfactual
message_prefix = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prefix},
    {"role": "assistant", "content": output_factual_text},
]
sampling_params = vllm.SamplingParams(temperature=0.1, seed=0)
outputs_cf = llm.generate([message_prefix], sampling_params)
output_cf_text = outputs_cf[0].outputs[0].text

print("========== Counterfactual ==========")
print(output_cf_text)

# interventional
sampling_params = vllm.SamplingParams(temperature=0.1, seed=1)
outputs_interventional = llm.generate([message_prefix], sampling_params)
output_interventional_text = outputs_interventional[0].outputs[0].text
print("========== Interventional ==========")
print(output_interventional_text)





