import os
import json
import requests
import openai
from tqdm import tqdm
import argparse
from datasets import load_from_disk


TITLE = "The Chronicles of Eldoria: The Forgotten Kingdom"
SCENARIO = "In the mystical land of Eldoria, where magic intertwines with the threads of reality, you find yourself waking up in a small, dimly lit hut. The scent of damp wood and old parchment fills the air. You are an apprentice to the old sage, Eledor, who has mysteriously disappeared. The only clue to his whereabouts is a cryptic note left on his desk, reading, \"Seek the Forgotten Kingdom, where shadows whisper secrets.\"\n\nYou step outside the hut, and the village of Elden greets you. The cobblestone paths, the thatched-roof houses, and the distant sound of the blacksmith's hammer hitting the anvil are all too familiar. Yet, an air of unease hangs over the village. The villagers, once cheerful and bustling with life, now wear worried expressions. Eledor's disappearance has not gone unnoticed.\n\nTo the north lies the dense Eldwood Forest, a place of enchantment and danger, where mythical creatures lurk in the shadows. To the east, the towering Eldoria Mountains stand guard, their peaks lost in the clouds. To the west, the tranquil Elden Lake mirrors the sky, hiding ancient secrets beneath its surface. And to the south, the vast Eldoria Plains stretch out, leading to unknown lands.\n\nYour adventure begins here, in the heart of Eldoria. Will you brave the Eldwood Forest, scale the Eldoria Mountains, dive into the depths of Elden Lake, or traverse the Eldoria Plains? The choice is yours. Remember, the fate of Eledor and the entire village of Elden rests on your shoulders. Seek the Forgotten Kingdom, decipher its secrets, and save your mentor. The Chronicles of Eldoria await you."


def run_status(outcome_dataset_path, uri, **kwargs):
    
    dataset = load_from_disk(outcome_dataset_path)
    
    for k, v in kwargs.items():
        print(k+':'+str(v))

    openai.organization = ""                        # Don't leak this
    openai.api_key = ""      # Don't leak this
    
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer {}'.format(openai.api_key)}
    
    results = []

    print("Sending Requests to OpenAI...")
    
    for _, data in enumerate(tqdm(dataset)):
        prompt = data['result']        
        data = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful text-based adventure game assistant. Given the result of the player's action, classify the outcome of it into one of the 'Dead', 'Win', 'Wounded', 'Healed', and 'Nothing happened'. Dead: the character is dead in the result.\n\nWin: the result indicates that the player has achieved the goal of the quest: save his mentor.\n\nWounded: the player is injured\n\nHealed: the player is healed\n\nNothing happened: everything stays the same.\n\nReturn and only return the classified category."},
                    {"role": "user", "content": prompt}
                ],
                **kwargs
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

def main():
    params = {
        "model": "gpt-3.5-turbo",
        "n": 1,
        "temperature": 0.3,
        "max_tokens": 256,
    }

    uri = "https://api.openai.com/v1/chat/completions"
    outcome_dataset_path = "./processed_data_v1.2/action_outcome"
    json.dump(run_status(outcome_dataset_path, uri, **params), open("status.json", "w"))
    

if __name__ == "__main__":
    main()
