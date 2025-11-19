import config_env
import os
        
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from datasets import  load_dataset
import re
from model_load import load_model, model_name_func

def generate_MAWPS(data):
        question, ans, numbers = data['Question'], data['Answer'], data['Numbers']
        numbers = data['Numbers'].split()
        for i, number in enumerate(numbers):
            # print(number)
            placeholder = f'N_{i:02d}'
            question = re.sub(rf'\b{placeholder}\b', number, question)
            prompt = "Question: " + str(question) + "\nLet's think step by step\nAnswer:\n"
            max_token = 200
        return question, prompt, ans, max_token

if __name__ == '__main__':
    name = 'Qwen3-0.6B'
    dataset_name = 'maw'
    model_name =  model_name_func(name)# 'meta-llama/Llama-3.2-1B'# "Qwen/Qwen2.5-1.5B"
    dataset = load_dataset("mwpt5/MAWPS")
    test = dataset["train"]
    print(dataset['train'][0])
    print(len(test))
    model = load_model(name)
    print(model.cfg.n_layers)

    save_json = f'./data_{dataset_name}/'
    text = []
    
    for idx,data in enumerate(test):
        question, prompt, ans, max_token = generate_MAWPS(data)
        generated_text = model.generate(
            prompt,
            max_new_tokens=max_token,  
            temperature=0,     
        )
        # print(generated_text, ans)
        text.append({'question': question, 'gold_ans': ans, 'ans':generated_text})
        print(idx)

    if os.path.exists(save_json) == False:
        os.makedirs(save_json)

    with open(save_json + f'{name}_answer.json', 'w') as f:
        json.dump(text, f)

    # print(model)