import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pytorch_lightning as pl
import argparse
from pytorch_lightning import Trainer, seed_everything
from modules import ActionDataModule, ResultDataModule, Model
import os
import json
from torchmetrics.text.rouge import ROUGEScore
import requests


def chatgpt_eval(batch, tokenizer, action_or_result="result"):
    
    params = {
        "model": "gpt-3.5-turbo",
        "n": 1,
        "temperature": 0.3,
        "max_tokens": 256,
    }

    uri = "https://api.openai.com/v1/chat/completions"
    openai.organization = ""                        # Don't leak this
    openai.api_key = ""      # Don't leak this
    
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer {}'.format(openai.api_key)}
    
    results = []

    print("Sending Requests to OpenAI...")
    
    if action_or_result == "result":
        content = "You are a helpful text-based adventure game designer. Given the title, the scenario, and the action taken, generate the result of the action and the new situation. Return in the format 'Result: <result>\n\nNew Situation: <new situation>"
    else:
        content = "You are a helpful text-based adventure game designer. Given the title, background, and current situation, generate a random valid action (in one short sentence) for the player to take. Return in the format 'Action: <action>'."
    
    for input_id, mask in zip(batch["input_ids"], batch["attention_mask"]):
        input_id = [input_id[i] for i in range(len(input_id)) if mask[i] == 1]
        input_text = tokenizer.decode(input_id, skip_special_tokens=True)
              
        data = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": content},
                    {"role": "user", "content": input_text}
                ],
                **params
            })
        try:
            response = requests.post(uri, headers=headers, data=data).json()
            result = response['choices'][0]['message']['content']
        except:
            import time
            print("Sleep for 60s")
            time.sleep(300)
            
            response = requests.post(uri, headers=headers, data=data).json()
            result = response['choices'][0]['message']['content']
            
        results.append(result)
        
    return results
    
    
    pass

def eval(args):
    
    tokenizer = AutoTokenizer.from_pretrained("/work/frink/models/llama3-8B-Instruct-HF")
    llama_baseline_model = AutoModelForCausalLM.from_pretrained("/work/frink/models/llama3-8B-Instruct-HF")
    llama_baseline_model = llama_baseline_model.to("cuda")
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    torch.set_float32_matmul_precision("high")

    result_datamodule = ResultDataModule(args, tokenizer)
    action_datamodule = ActionDataModule(args, tokenizer)
    
    result_eval_data = result_datamodule.val_dataloader()
    action_eval_data = action_datamodule.val_dataloader()
    
    result_llama_prediction = []
    result_gpt_prediction = []
    result_labels = []
    
    result_llama_rouge = []
    result_gpt_rouge = []
    
    action_llama_prediction = []
    action_gpt_prediction = []
    action_labels = []
    
    action_llama_rouge = []
    action_gpt_rouge = []
    
    rouge = ROUGEScore()
        
    for batch in result_eval_data:
        
        batch = {k: v.to("cuda") for k, v in batch.items()}
        
        labels = [[label[i].item() for i in range(len(label)) if label[i].item() != -100] for label in batch["labels"]]
        result_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
        
        llama_pred = llama_baseline_model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_length=1024)
        llama_pred = llama_pred[:, batch["input_ids"].shape[-1]:]
        llama_pred = tokenizer.batch_decode(llama_pred, skip_special_tokens=True)
        result_llama_prediction.extend(llama_pred)
        gpt_pred = chatgpt_eval(batch, tokenizer, action_or_result="result")
        result_gpt_prediction.extend(gpt_pred)
    
    for llama, gpt, gold in zip(result_llama_prediction, result_gpt_prediction, result_labels):
        llama_rouge = rouge(gpt, gold)["rouge2_fmeasure"]
        gpt_rouge = rouge(llama, gold)["rouge2_fmeasure"]
        result_llama_rouge.append(llama_rouge)
        result_gpt_rouge.append(gpt_rouge)
    
    result_llama_rouge = torch.stack(result_llama_rouge).mean()
    result_gpt_rouge = torch.stack(result_gpt_rouge).mean()
    
    for batch in action_eval_data:
        
        batch = {k: v.to("cuda") for k, v in batch.items()}
        
        labels = [[label[i].item() for i in range(len(label)) if label[i].item() != -100] for label in batch["labels"]]
        action_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
        
        llama_pred = llama_baseline_model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_length=1024)
        llama_pred = llama_pred[:, batch["input_ids"].shape[-1]:]
        llama_pred = tokenizer.batch_decode(llama_pred, skip_special_tokens=True)
        action_llama_prediction.extend(llama_pred)
        gpt_pred = chatgpt_eval(batch, tokenizer, action_or_result="action")
        action_gpt_prediction.extend(gpt_pred)
        
    for llama, gpt, gold in zip(action_llama_prediction, action_gpt_prediction, action_labels):
        llama_rouge = rouge(gpt, gold)["rouge2_fmeasure"]
        gpt_rouge = rouge(llama, gold)["rouge2_fmeasure"]
        action_llama_rouge.append(llama_rouge)
        action_gpt_rouge.append(gpt_rouge)
        
    action_llama_rouge = torch.stack(action_llama_rouge).mean()
    action_gpt_rouge = torch.stack(action_gpt_rouge).mean()
    
    results = {
        "result_llama_rouge": result_llama_rouge,
        "result_gpt_rouge": result_gpt_rouge,
        "action_llama_rouge": action_llama_rouge,
        "action_gpt_rouge": action_gpt_rouge
    }
    
    with open(args.output_dir, "w") as f:
        json.dump(results, f)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./baseline_results.json")
    parser.add_argument("--input_dir", type=str, default="./processed_data_v1.2")
    parser.add_argument("--devices", type=list, default=[0])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_set", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", type=bool, default=True)
    args = parser.parse_args()
    seed_everything(args.seed)
    eval(args)
