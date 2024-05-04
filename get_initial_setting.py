import os
import json
import requests
import openai
from tqdm import tqdm


def run(inputs, output_dir, uri, **kwargs):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    json.dump(kwargs, open(os.path.join(output_dir, "params.json"), "w"))
    for k, v in kwargs.items():
        print(k+':'+str(v))

    openai.organization = "" # Don't leak this
    openai.api_key = ""  # Don't leak this
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer {}'.format(openai.api_key)}

    print("Sending Requests to OpenAI...")
    for idx, prompt in tqdm(enumerate(inputs)):
        if uri == "https://api.openai.com/v1/chat/completions":
            data = json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful game designer"},
                        {"role": "user", "content": prompt}
                    ],
                    **kwargs
                })
        else:
            data = json.dumps({"prompt": prompt, **kwargs})
        response = requests.post(uri, headers=headers, data=data).json()
        
        json.dump(response.json(), open(os.path.join(output_dir, "result_{}.json".format(idx + 1)), "w"))


def main():
    params = {
        "model": "gpt-4",
        "n": 10,
        "temperature": 0.3,
        "max_tokens": 512,
    }

    uri = "https://api.openai.com/v1/chat/completions"

    prompts = [
        "Please help me to create a starter scenario for a classic fantasy text-based adventure game. Return nothing but the title and the content of the initial scenario",
    ]

    run(prompts, os.path.join("data", "initial_scenario"), uri, **params)


if __name__ == "__main__":
    main()
