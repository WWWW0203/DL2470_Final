from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, RobertaForSequenceClassification
import transformers
import torch
import sys,time,random
from transformers.utils import logging
import random


typing_speed = 150 #wpm
def slow_type(t, typing_speed=typing_speed):
    for l in t:
        sys.stdout.write(l)
        sys.stdout.flush()
        time.sleep(random.random()*10.0/typing_speed)
    print()
    
logging.set_verbosity(transformers.logging.FATAL)
STATUS_SPACE = ["♥♥", "♥♡"]

TITLE = "The Chronicles of Eldoria: The Forgotten Kingdom"
SCENARIO = "In the mystical land of Eldoria, where magic intertwines with the threads of reality, you find yourself waking up in a small, dimly lit hut. The scent of damp wood and old parchment fills the air. You are an apprentice to the old sage, Eledor, who has mysteriously disappeared. The only clue to his whereabouts is a cryptic note left on his desk, reading, \"Seek the Forgotten Kingdom, where shadows whisper secrets.\"\n\nYou step outside the hut, and the village of Elden greets you. The cobblestone paths, the thatched-roof houses, and the distant sound of the blacksmith's hammer hitting the anvil are all too familiar. Yet, an air of unease hangs over the village. The villagers, once cheerful and bustling with life, now wear worried expressions. Eledor's disappearance has not gone unnoticed.\n\nTo the north lies the dense Eldwood Forest, a place of enchantment and danger, where mythical creatures lurk in the shadows. To the east, the towering Eldoria Mountains stand guard, their peaks lost in the clouds. To the west, the tranquil Elden Lake mirrors the sky, hiding ancient secrets beneath its surface. And to the south, the vast Eldoria Plains stretch out, leading to unknown lands.\n\nYour adventure begins here, in the heart of Eldoria. Will you brave the Eldwood Forest, scale the Eldoria Mountains, dive into the depths of Elden Lake, or traverse the Eldoria Plains? The choice is yours. Remember, the fate of Eledor and the entire village of Elden rests on your shoulders. Seek the Forgotten Kingdom, decipher its secrets, and save your mentor. The Chronicles of Eldoria await you."

slow_type("Loading the world...")
outcome_model = AutoPeftModelForCausalLM.from_pretrained("model/result_model/llama3-8B-Instruct-HF", torch_dtype=torch.bfloat16)
action_model = AutoPeftModelForCausalLM.from_pretrained("model/action_model/llama3-8B-Instruct-HF", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("/work/frink/models/llama3-8B-Instruct-HF")


roberta_model = RobertaForSequenceClassification.from_pretrained("model/status_model/xlm-roberta-large")
roberta_tokenizer = AutoTokenizer.from_pretrained("model/status_model/xlm-roberta-large")


def action_prompter(model, tokenizer, title, background, cur_situation):
    input_text = f"Title: {title}\n\nBackground: {background}\n\nCurrent Situation: {cur_situation}\n\nAction: "
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    output = model.generate(input_ids, max_length=512, pad_token_id=tokenizer.eos_token_id)
    output = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(output, skip_special_tokens=True)

def game_controller(model, tokenizer, title, background, cur_situation, action):
    input_text = f"Title: {title}\n\nBackground: {background}\n\nCurrent Situation: {cur_situation}\n\nAction: {action}\n\n"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    output = model.generate(input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id)
    output = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(output, skip_special_tokens=True)


def classify_status_change(model, tokenizer, result):
    input_ids = tokenizer.encode(result, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    logits = model(input_ids).logits
    return logits.argmax().item()


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.pad_token_id = tokenizer.eos_token_id

outcome_model = outcome_model.to("cuda")
outcome_model.eval()

action_model = action_model.to("cuda")
action_model.eval()

roberta_model = roberta_model.to("cuda")
roberta_model.eval()


status = "♥♥"

slow_type("Hi, welcome to text-based fantasy game: The Chronicles of Eldoria: The Forgotten Kingdom!")
print()
time.sleep(0.25)
print()
time.sleep(0.25)
slow_type(SCENARIO, typing_speed=600)

cur_situation = ''
cur_result = ''

while True:
    user_input = input("What are you going to do?\nEnter your next move: ")
    
    if user_input == "exit":
        pass
    elif user_input == "Help me!":
        action = action_prompter(action_model, tokenizer, TITLE, SCENARIO, cur_situation)
        slow_type(f"Here's something that you can do: {action}", typing_speed=300)  
        user_input = ""
        continue
    elif user_input == "exit":
        break
    else:
        outcome = game_controller(outcome_model, tokenizer, TITLE, SCENARIO, cur_situation, user_input)
        result = outcome.split("Result: ")[1].split("New Situation: ")[0].strip()
        new_situation = outcome.split("New Situation: ")[1].strip()
        
        status_change = classify_status_change(roberta_model, roberta_tokenizer, result)
        
        slow_type(f"{result}", typing_speed=600)
        
        if status_change == 0:
            if random.random() < 0.2:
                slow_type("You have won the game!")
                break
        elif status_change == 1:
            slow_type("You have lost the game!")
            break
        elif status_change == 2:
            if status == "♥♥":
                slow_type("You are injured!")
                status = "♥♡"
            else:
                slow_type("You have lost the game!")
                break
        elif status_change == 3:
            if status == "♥♡":
                slow_type("You have recovered!")
                status = "♥♥"
        else:
            pass
        
        slow_type(f"Your current status: {status}\n")
        slow_type(f"Your current quest: 1. Save your mentor\n")
        slow_type(new_situation, typing_speed=600)
        cur_situation = new_situation
    
print("Game over!")