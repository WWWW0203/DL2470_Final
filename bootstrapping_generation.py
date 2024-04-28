from spawn_valid_actions import run_actions
from spawn_valid_outcomes import run_outcomes
import os
import json
import argparse
import tqdm

def step(actions, outcomes, depth, max_depth, save_dir, action_per_scenario):
    assert len(actions) == len(outcomes)
    
    uri = "https://api.openai.com/v1/chat/completions"
    
    if depth == max_depth:
        return
    
    for idx, outcome in enumerate(outcomes):
        cur_situation = outcome["new_situation"]
        print(f"Step {depth} - Scenario {idx+1}")
        
        action_params = {
            "model": "gpt-4",
            "n": 1,
            "temperature": 0.3,
            "max_tokens": 128,
        }
        
        outcome_params = {
            "model": "gpt-4",
            "n": 1,
            "temperature": 0.3,
            "max_tokens": 256,
        }

        new_save_dir = os.path.join(save_dir, f"scenario_{idx+1}")
        if not os.path.exists(new_save_dir):
            os.makedirs(new_save_dir, exist_ok=True)
                        
        new_actions = run_actions(cur_situation, uri, num_run=action_per_scenario, **action_params)
        
        new_outcomes = []
        for new_action in new_actions:
            new_outcome = run_outcomes(cur_situation, new_action, uri, **outcome_params)
            new_outcomes.append(new_outcome)
        
        json.dump(new_actions, open(os.path.join(new_save_dir, "actions.json"), "w"))
        json.dump(new_outcomes, open(os.path.join(new_save_dir, "outcomes.json"), "w"))
        
        step(new_actions, new_outcomes, depth+1, max_depth, new_save_dir, action_per_scenario)


def bootstrap(args):
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    initial_actions = json.load(open(args.initial_actions, "r"))
    initial_outcomes = json.load(open(args.initial_outcomes, "r"))
    assert len(initial_actions) == len(initial_outcomes)
    step(initial_actions, initial_outcomes, 0, args.max_depth, args.output_dir, args.actions_per_scenario)
    


def main():
    parser = argparse.ArgumentParser(description='Generate valid initial outcomes for a given scenario')
    parser.add_argument('--actions_per_scenario', type=int, default=2)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--output_dir', type=str, default="data")
    parser.add_argument('--initial_actions', type=str, default="data/actions.json")
    parser.add_argument('--initial_outcomes', type=str, default="data/outcomes.json")
    
    args = parser.parse_args()
    
    bootstrap(args)
    
    

if __name__ == "__main__":
    main()
    

