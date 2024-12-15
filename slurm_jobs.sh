conda activate pistol

model_families=("llama2-7b-chat" "gemma-7b-it" "llama3-8b-instruct" "Qwen2-7B-Instruct")
pretrained_model_paths=("meta-llama/Llama-2-7b-chat-hf" "google/gemma-7b-it" "meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen2-7B-Instruct")
num_runs=(1 2 3)

dataset_names=("tofu_full" "pistol_sample1")
data_paths=("data/tofu_full.json" "data/sample_data_1.json")
forget_types=("author1" "forget_AB")
forget_edges=("[\"author_1\"]" "[\"A_B\"]")

pistol_lrs=(1.5e-5 1.5e-5 5e-6 1.5e-5)
tofu_lrs=(2e-5 2e-5 1e-5 1.5e-5)

for j in "${!dataset_names[@]}"; do
  dataset_name=${dataset_names[$j]}
  data_path=${data_paths[$j]}
  forget_type=${forget_types[$j]}
  forget_edge=${forget_edges[$j]}
  
  if [ "$dataset_name" == "pistol_sample1" ]; then
    lrs=("${pistol_lrs[@]}")
  elif [ "$dataset_name" == "tofu_full" ]; then
    lrs=("${tofu_lrs[@]}")
  fi
  
  for i in "${!model_families[@]}"; do
    model_family=${model_families[$i]}
    pretrained_model_path=${pretrained_model_paths[$i]}
    lr=${lrs[$i]}
    
    for num_run in "${num_runs[@]}"; do
      job_name="${dataset_name}-eval-${model_family}-${num_run}"
      sbatch <<EOT
#!/bin/bash
#SBATCH -c 10
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --job-name=${job_name}
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=10:00:00

export HF_TOKEN=\$(cat /nfs-share/mk2296/.huggingface_token)

which python
python eval.py forget.num_run=${num_run} forget.lr=${lr} model_family='${model_family}' pretrained_model_path='${pretrained_model_path}' forget.num_run=${num_run} dataset_name='${dataset_name}' data_path='${data_path}' forget_type='${forget_type}' forget_edge='${forget_edge}'
EOT
    done
  done
done