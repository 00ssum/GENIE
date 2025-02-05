import os
import re
import pandas as pd

def extract_log_data(log_file_path):
    """Extract required values from the log file."""
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    test_domain_validation = re.search(r"test-domain validation\(oracle\) = ([\d.]+)%", content)
    training_domain_validation = re.search(r"training-domain validation\(iid\) = ([\d.]+)%", content)
    last_value = re.search(r"last = ([\d.]+)%", content)
    
    trial_seed_match = re.search(r"--trial_seed (\d+)", content)
    hparams_seed_match = re.search(r"--hparams_seed (\d+)", content)
    if not trial_seed_match:
        trial_seed_match = re.search(r"trial_seed:\s+(\d+)", content)  # Args 섹션에서 찾기
        hparams_seed_match = re.search(r"--hparams_seed (\d+)", content)
    if not hparams_seed_match:
        hparams_seed_match = re.search(r"hparams_seed:\s+(\d+)", content)  # Args 섹션에서 찾기
    
    return {
        "test-domain validation(oracle)": float(test_domain_validation.group(1)) if test_domain_validation else None,
        "training-domain validation(iid)": float(training_domain_validation.group(1)) if training_domain_validation else None,
        "last": float(last_value.group(1)) if last_value else None,
        "trial_seed": int(trial_seed_match.group(1)) if trial_seed_match else None,
        "hparams_seed": int(hparams_seed_match.group(1)) if hparams_seed_match else None,
    }

def collect_results_by_dataset_and_algorithm(base_path, algorithms, datasets):
    """Walk through the directory structure and collect results for each dataset and algorithm."""
    results_by_dataset_and_algorithm = {}
    for dataset in datasets:
        dataset_results = {}
        for algorithm in algorithms:
            algorithm_path = os.path.join(base_path, dataset, algorithm)
            if not os.path.exists(algorithm_path):
                continue
            
            target_env_results = {}
            for target_env in os.listdir(algorithm_path):
                target_env_path = os.path.join(algorithm_path, target_env)
                if not os.path.isdir(target_env_path):
                    continue
                
                # 폴더 이름에서 대괄호 제거 및 쉼표와 공백을 언더스코어로 대체
                clean_target_env = target_env.strip("[]").replace(", ", "_")

                results = []
                for folder_name in os.listdir(target_env_path):
                    if folder_name == "runs":  # Skip "runs" folder
                        continue
                    
                    folder_path = os.path.join(target_env_path, folder_name)
                    log_file_path = os.path.join(folder_path, "log.txt")
                    if os.path.exists(log_file_path):
                        log_data = extract_log_data(log_file_path)
                        results.append({
                            "algorithm": algorithm,
                            "target_env": clean_target_env,
                            "folder": folder_name,
                            **log_data
                        })
                if results:
                    target_env_results[clean_target_env] = results
            if target_env_results:
                dataset_results[algorithm] = target_env_results
        if dataset_results:
            results_by_dataset_and_algorithm[dataset] = dataset_results
    return results_by_dataset_and_algorithm

def save_to_excel_by_dataset_algorithm_and_env(results_by_dataset_and_algorithm, output_dir):
    """Save results to separate Excel files for each dataset, algorithm, and target_env."""
    os.makedirs(output_dir, exist_ok=True)
    for dataset, algorithm_results in results_by_dataset_and_algorithm.items():
        dataset_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        for algorithm, env_results in algorithm_results.items():
            algorithm_dir = os.path.join(dataset_dir, algorithm)
            os.makedirs(algorithm_dir, exist_ok=True)
            for target_env, results in env_results.items():
                # target_env에 _이 포함되었으면 SDG, 포함되지 않으면 DG
                subfolder = "SDG" if "_" in target_env else "DG"

                output_file = os.path.join(algorithm_dir, subfolder, f"{target_env}.xlsx")
                os.makedirs(os.path.join(algorithm_dir, subfolder), exist_ok=True)

                df = pd.DataFrame(results, columns=[
                    "algorithm", "target_env", "folder",
                    "test-domain validation(oracle)", 
                    "training-domain validation(iid)", 
                    "last",
                    "trial_seed",
                    "hparams_seed"
                ])
                df.to_excel(output_file, index=False)
                print(f"Saved {dataset} -> {algorithm} -> {subfolder or 'default'} -> target_env {target_env} results to {output_file}")

# Base path to the dataset directory
base_path = "/jsm0707/GENIE/train_output"
output_dir = "output/results"
algorithms = ["gsnr1224", "CORAL", "GENIE", "ERM", "SAM","RSC", "MIRO"]
datasets = ["PACS", "OfficeHome", "TerraIncognita", "VLCS"]

# Collect results and save to Excel
results_by_dataset_and_algorithm = collect_results_by_dataset_and_algorithm(base_path, algorithms, datasets)
save_to_excel_by_dataset_algorithm_and_env(results_by_dataset_and_algorithm, output_dir)
