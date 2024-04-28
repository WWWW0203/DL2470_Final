import os
import json
import requests
import openai
from tqdm import tqdm
import argparse


TITLE = "The Chronicles of Eldoria: The Forgotten Kingdom"
SCENARIO = "In the mystical land of Eldoria, where magic intertwines with the threads of reality, you find yourself waking up in a small, dimly lit hut. The scent of damp wood and old parchment fills the air. You are an apprentice to the old sage, Eledor, who has mysteriously disappeared. The only clue to his whereabouts is a cryptic note left on his desk, reading, \"Seek the Forgotten Kingdom, where shadows whisper secrets.\"\n\nYou step outside the hut, and the village of Elden greets you. The cobblestone paths, the thatched-roof houses, and the distant sound of the blacksmith's hammer hitting the anvil are all too familiar. Yet, an air of unease hangs over the village. The villagers, once cheerful and bustling with life, now wear worried expressions. Eledor's disappearance has not gone unnoticed.\n\nTo the north lies the dense Eldwood Forest, a place of enchantment and danger, where mythical creatures lurk in the shadows. To the east, the towering Eldoria Mountains stand guard, their peaks lost in the clouds. To the west, the tranquil Elden Lake mirrors the sky, hiding ancient secrets beneath its surface. And to the south, the vast Eldoria Plains stretch out, leading to unknown lands.\n\nYour adventure begins here, in the heart of Eldoria. Will you brave the Eldwood Forest, scale the Eldoria Mountains, dive into the depths of Elden Lake, or traverse the Eldoria Plains? The choice is yours. Remember, the fate of Eledor and the entire village of Elden rests on your shoulders. Seek the Forgotten Kingdom, decipher its secrets, and save your mentor. The Chronicles of Eldoria await you."


def run_actions(cur_situation, uri, num_run, **kwargs):
    
    prompt = f"Title: {TITLE}\n\Background: {SCENARIO}\n\nCurrent Situation: {cur_situation}"

    for k, v in kwargs.items():
        print(k+':'+str(v))

    openai.organization = ""                        # Don't leak this
    openai.api_key = ""      # Don't leak this
    
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer {}'.format(openai.api_key)}
    
    results = []

    print("Sending Requests to OpenAI...")
    data = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful text-based adventure game designer. Given the title, background, and current situation, generate a random valid action (in one short sentence) for the player to take. Return in the format 'Action: <action>'."},
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
    
    msg = [
        {"role": "system", "content": "You are a helpful text-based adventure game designer. Given the title and the scenario, generate a random valid action (in one short sentence) for the player to take. Return in the format 'Action: <action>'."},
        {"role": "user", "content": prompt}
    ]
    
    while len(results) < num_run:
        msg.append({"role": "system", "content": results[-1]})
        msg.append({"role": "user", "content": "Come up with another different action."})
        
        data = json.dumps(
            {
                "messages": msg,
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
    
    results = [result.split("Action: ")[1] for result in results]
        
    return results

def main():
    params = {
        "model": "gpt-4",
        "n": 1,
        "temperature": 0.3,
        "max_tokens": 128,
    }

    uri = "https://api.openai.com/v1/chat/completions"
    cur_situation = "You find yourself in the heart of the Eldwood Forest. The path ahead is shrouded in shadows, and every now and then, you hear a rustle in the undergrowth. The forest is alive with magic, and you can feel it tingling on your skin. You have a choice to make: will you follow the path that leads deeper into the forest, or will you veer off the beaten path and explore the unknown? The fate of Eledor and Elden village depends on your decision."

    run_actions(cur_situation, uri, num_run=10, **params)


if __name__ == "__main__":
    main()
