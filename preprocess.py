import os
import json
import datasets


TITLE = "The Chronicles of Eldoria: The Forgotten Kingdom"
SCENARIO = "In the mystical land of Eldoria, where magic intertwines with the threads of reality, you find yourself waking up in a small, dimly lit hut. The scent of damp wood and old parchment fills the air. You are an apprentice to the old sage, Eledor, who has mysteriously disappeared. The only clue to his whereabouts is a cryptic note left on his desk, reading, \"Seek the Forgotten Kingdom, where shadows whisper secrets.\"\n\nYou step outside the hut, and the village of Elden greets you. The cobblestone paths, the thatched-roof houses, and the distant sound of the blacksmith's hammer hitting the anvil are all too familiar. Yet, an air of unease hangs over the village. The villagers, once cheerful and bustling with life, now wear worried expressions. Eledor's disappearance has not gone unnoticed.\n\nTo the north lies the dense Eldwood Forest, a place of enchantment and danger, where mythical creatures lurk in the shadows. To the east, the towering Eldoria Mountains stand guard, their peaks lost in the clouds. To the west, the tranquil Elden Lake mirrors the sky, hiding ancient secrets beneath its surface. And to the south, the vast Eldoria Plains stretch out, leading to unknown lands.\n\nYour adventure begins here, in the heart of Eldoria. Will you brave the Eldwood Forest, scale the Eldoria Mountains, dive into the depths of Elden Lake, or traverse the Eldoria Plains? The choice is yours. Remember, the fate of Eledor and the entire village of Elden rests on your shoulders. Seek the Forgotten Kingdom, decipher its secrets, and save your mentor. The Chronicles of Eldoria await you."


def _get_outcome_action_pairs(cur_dir, cur_situation):
    actions = json.load(open(os.path.join(cur_dir, "actions.json"), "r"))
    outcomes = json.load(open(os.path.join(cur_dir, "outcomes.json"), "r"))
    
    action_outcome_pairs = []
    for i, action in enumerate(actions):
        action_outcome_pairs.append({"cur_situation": cur_situation, "action": action})
        
        next_dir = os.path.join(cur_dir, f"scenario_{i+1}")
        if os.path.exists(next_dir):
            action_outcome_pairs.extend(_get_outcome_action_pairs(next_dir, outcomes[i]["new_situation"]))
            
    return action_outcome_pairs


def _get_action_outcome_pairs(cur_dir, cur_situation):
    actions = json.load(open(os.path.join(cur_dir, "actions.json"), "r"))
    outcomes = json.load(open(os.path.join(cur_dir, "outcomes.json"), "r"))
    
    action_outcome_pairs = []
    for action, outcome in zip(actions, outcomes):
        result = outcome["result"]
        situation = outcome["new_situation"]
        action_outcome_pairs.append({"cur_situation": cur_situation, "action": action, "result": result, "new_situation": situation})
    
    
    for i, outcome in enumerate(outcomes):
        next_dir = os.path.join(cur_dir, f"scenario_{i+1}")
        if os.path.exists(next_dir):
            action_outcome_pairs.extend(_get_action_outcome_pairs(next_dir, outcome["new_situation"]))
        
    return action_outcome_pairs


def preprocess_action_outcome_pairs(data_dir):
        
    action_outcome_dataset = _get_action_outcome_pairs(data_dir, "")
    action_outcome_dataset = datasets.Dataset.from_list(action_outcome_dataset)
    
    return action_outcome_dataset


def process_outcome_action_pairs(data_dir):
    
    outcome_action_dataset = _get_outcome_action_pairs(data_dir, "")
    outcome_action_dataset = datasets.Dataset.from_list(outcome_action_dataset)
    
    return outcome_action_dataset

    
    
if __name__ == "__main__":
    action_outcome = preprocess_action_outcome_pairs("./data")
    outcome_action = process_outcome_action_pairs("./data")
    
    print(len(action_outcome), len(outcome_action))
    if not os.path.exists("processed_data"):
        os.makedirs("processed_data", exist_ok=True)
    
    action_outcome.save_to_disk("processed_data/action_outcome")
    outcome_action.save_to_disk("processed_data/outcome_action")
    