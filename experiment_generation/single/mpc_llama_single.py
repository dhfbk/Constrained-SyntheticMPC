import pickle
import json
from vllm import LLM, SamplingParams
from huggingface_hub import login
login("YOUR_TOKEN")

input_path = "input_path"
output_path = "output_path"

#open json file "prompts.json" and load the data
with open(input_path, "r") as f:
    prompts = json.load(f)

llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", seed=42)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, top_k=40, max_tokens=4576)

count = 1
list_outp = []
gen_outp = []

for prompt in prompts["prompts"]:
    instruction = prompt["text"]

    messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": "Generate a conversation."},
        ]
    
    conversations = [messages for _ in range(75)]        
    print(count)
    count += 1

    outputs = llm.chat(messages=conversations,
                       sampling_params=sampling_params,
                       use_tqdm=True)

    for o in outputs:
        generated_text = o.outputs[0].text
        list_outp.append(generated_text)

with open(output_path, 'wb') as f:
    pickle.dump(list_outp, f)
