import os
import json
import hydra 
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

def load_idk_json():
    idk_path = 'data/refusal_response.txt'
    with open(idk_path, "r") as file:
        responses = [line.strip() for line in file if line.strip()]
    return responses

def get_similarity(model, tokenizer, response1, response2):
    # Existing similarity calculation function remains unchanged
    inputs1 = tokenizer(response1, return_tensors="pt", truncation=True, padding=True)
    inputs2 = tokenizer(response2, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs1 = model(**inputs1,output_hidden_states=True)
        outputs2 = model(**inputs2,output_hidden_states=True)

    hidden_states1 = outputs1.hidden_states[-1]
    hidden_states2 = outputs2.hidden_states[-1]

    sentence_embedding1 = hidden_states1.mean(dim=1).squeeze(0)
    sentence_embedding2 = hidden_states2.mean(dim=1).squeeze(0)

    embedding1 = sentence_embedding1.to(dtype=torch.float32).cpu().numpy()
    embedding2 = sentence_embedding2.to(dtype=torch.float32).cpu().numpy()

    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # Load model and tokenizer
    pretrained_model = 'meta-llama/Meta-Llama-3-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model,
        use_flash_attention_2=False,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    base_path = "eval_forget"#cfg.res.base_path
    metrics = ["rouge1_recall", "mrr", "hit_rate"]
    eval_types = ["forget", "all_retained_edge", "factual_data"]
    results = []

    # Iterate through the directory structure
    for model_family in os.listdir(base_path):
        model_family_path = os.path.join(base_path, model_family)
        if not os.path.isdir(model_family_path):
            continue

        for forget_set in os.listdir(model_family_path):
            forget_set_path = os.path.join(model_family_path, forget_set)
            if not os.path.isdir(forget_set_path):
                continue

            for baseline in os.listdir(forget_set_path):
                baseline_path = os.path.join(forget_set_path, baseline)
                if not os.path.isdir(baseline_path):
                    continue

                # Initialize metric storage for this configuration
                metric_values = {eval_type: {metric: [] for metric in metrics} for eval_type in eval_types}
                refusal_similarities = []

                # Iterate through run IDs (1, 2, 3)
                for run_id in ["1", "2", "3"]:
                    run_path = os.path.join(baseline_path, run_id)
                    if not os.path.exists(run_path):
                        continue

                    # Get the first subfolder (lr_config)
                    subfolders = [f for f in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, f))]
                    if not subfolders:
                        continue
                    
                    lr_config = subfolders[0]  # Take the first subfolder
                    
                    eval_path = os.path.join(run_path, lr_config, "eval.json")
                    #print (eval_path)

                    if not os.path.exists(eval_path):
                        continue

                    # Load and process evaluation data
                    with open(eval_path, "r") as file:
                        data = json.load(file)

                    # Collect metrics for each evaluation type
                    for eval_type in eval_types:
                        if eval_type in data:
                            for metric in metrics:
                                if metric in data[eval_type]:
                                    metric_values[eval_type][metric].append(data[eval_type][metric])

                    # Calculate refusal similarity if forget data exists
                    if "forget" in data and "generated_text" in data["forget"]:
                        forget_gen = [item[1] for item in data["forget"]["generated_text"]]
                        refusal_response = load_idk_json()
                        
                        run_similarities = []
                        for response1 in forget_gen:
                            for response2 in refusal_response:
                                similarity = get_similarity(model, tokenizer, response1, response2)
                                run_similarities.append(similarity)
                        
                        if run_similarities:
                            refusal_similarities.append(np.mean(run_similarities))

                # Calculate and store results
                for eval_type in eval_types:
                    for metric in metrics:
                        values = metric_values[eval_type][metric]
                        if values:
                            results.append({
                                "model_family": model_family,
                                "forget_set": forget_set,
                                "baseline": baseline,
                                "evaluation_type": eval_type,
                                "metric": metric,
                                "mean": np.mean(values),
                                "std": np.std(values),
                                "n_runs": len(values)
                            })

                # Add refusal similarity results if available
                if refusal_similarities:
                    results.append({
                        "model_family": model_family,
                        "forget_set": forget_set,
                        "baseline": baseline,
                        "evaluation_type": "forget",
                        "metric": "refusal_similarity",
                        "mean": np.mean(refusal_similarities),
                        "std": np.std(refusal_similarities),
                        "n_runs": len(refusal_similarities)
                    })

    # Create and save results DataFrame
    results_df = pd.DataFrame(results)
    output_path = os.path.join(".", "aggregated_eval_results.csv")
    results_df.to_csv(output_path, index=False)
    print(results_df)

if __name__ == "__main__":
    main()