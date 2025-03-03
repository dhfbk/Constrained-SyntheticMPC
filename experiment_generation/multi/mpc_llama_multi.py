import sys
import pickle
import json
from vllm import LLM, SamplingParams

from huggingface_hub import login
login("YOUR_TOKEN")

input_path = "input_path"
output_path = "output_path"


with open(input_path, "r") as f:
    prompts = json.load(f)

llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", seed=42)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, top_k=40, max_tokens=150)

tasks = ["Generate a speaker.",
         "Choose the next turn interactions.",
         "Generate the next message."
         ]

count = 1
list_outp = []
gen_outp = []

for prompt in prompts["prompts"]:
    print(count)
    count += 1
    instruction = prompt["text"]
    
    numb_user = prompt["total_user"]
        
    messages = [
            {"role": "system", "content": instruction}
        ]
    
    conversations = [messages for _ in range(75)] 
    
    speaker_prompt_basic = [
            {"role": "user", "content": tasks[0]}
        ]

    
    interaction_prompt_basic = [
            {"role": "user", "content": tasks[1]}
        ]

    
    text_prompt_basic = [
            {"role": "user", "content": tasks[2]}
        ]

    for us in range(numb_user):
        
        conversations= [conv + speaker_prompt_basic for conv in conversations]
        
        outputs = llm.chat(messages=conversations,
                       sampling_params=sampling_params,
                       use_tqdm=True)
        assistant_turn = [{"role": "assistant", "content": o.outputs[0].text} for o in outputs]
        
        conversations= [conv +  [assist] for conv, assist in zip(conversations, assistant_turn)]

    
    
    for round_turn in range(15):
        conversations= [conv + interaction_prompt_basic for conv in conversations]
        
        outputs = llm.chat(messages=conversations,
                       sampling_params=sampling_params,
                       use_tqdm=True)
        assistant_turn = [{"role": "assistant", "content": o.outputs[0].text} for o in outputs]
        
        print()
        
        conversations= [conv + [assist] for conv, assist in zip(conversations, assistant_turn)]

        conversations= [conv + text_prompt_basic for conv in conversations]
        
        outputs = llm.chat(messages=conversations,
                       sampling_params=sampling_params,
                       use_tqdm=True)
        
        assistant_turn = [{"role": "assistant", "content": o.outputs[0].text} for o in outputs]
        
        conversations= [conv +  [assist] for conv, assist in zip(conversations, assistant_turn)]

        
        
    list_outp = list_outp + conversations


with open(output_path, 'wb') as f:
    pickle.dump(list_outp, f)

