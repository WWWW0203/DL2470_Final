import json
import os
from datasets import load_from_disk, Dataset


if __name__ == "__main__":
    action_outcome = load_from_disk("processed_data_v1.2/action_outcome")
    outcome_action = load_from_disk("processed_data_v1.2/outcome_action")
    
    status = json.load(open("status.json", "r"))
    
    result_status_dataset = []
    for data, s in zip(action_outcome, status):
        
        result_status_dataset.append({
            "result": data["result"],
            "status": s
        })
    
    result_status_dataset = Dataset.from_list(result_status_dataset)
    result_status_dataset.save_to_disk("processed_data_v1.2/result_status")
    
    
