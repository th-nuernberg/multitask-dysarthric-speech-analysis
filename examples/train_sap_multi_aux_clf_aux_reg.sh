#!/bin/bash

stage=1
stop_stage=3

# Path to your prepared SAP dataset in parquet format
DATASET_PATH="/path/to/your/prepared_sap_data"

# Path to the SAPC manifests (reference files for evaluation)
SAPC_REF_PATH="/path/to/your/sapc_template/manifests"

# Path to the SAPC audio data (required for inference on a specific split)
SAPC_AUDIO_PATH="/path/to/your/sapc_template/data/processed"

# A unique name for your experiment. This will be used to create the output directory.
exp_tag="sap_multitask_finetune"

tasks=(
  asr
  monoloudness
  asr_monoloudness
  harsh
  asr_harsh
  consonants
  asr_consonants
  intelligibility
  asr_intelligibility
  naturalness
  asr_naturalness
)
weights=(
  0.01 0.14 0.05 0.15 0.05 0.15 0.05 0.15 0.05 0.15 0.05
)
aux_reg_cols=(intelligbility naturalness imprecise_consonants monoloudness harsh_voice)

epochs=6
batch_size_per_gpu=8
eval_batch_size_per_gpu=8
test_batch_size_per_gpu=16
learning_rate=4.0e-5
lr_scheduler="linear"
metric_for_best_model="macro_f1" # "wer" or "macro_f1"
clf_loss_wgt=0.3 # Weight for auxiliary classifier and regression losses
min_duration=1.0
max_duration=30.0

# Which dataset split to run inference on (e.g., "dev", "test"), depends on your preparation of the SAP data
inference_split="dev"
apply_vad=false
cleanup_transcripts=false

export HF_HOME="${HOME}/.cache/huggingface"
export WANDB_DIR="./wandb"
export WANDB_PROJECT="phi4_multimodal_sap"
export TOKENIZERS_PARALLELISM=false


venv_dir="./venv"
if [ ! -d "${venv_dir}" ]; then
  echo "Virtual environment '${venv_dir}' not found. Creating..."
  python3 -m venv "${venv_dir}"
  source "${venv_dir}/bin/activate"
  pip install --upgrade pip
  pip install wheel packaging
  pip install -r requirements.txt
  pip install flash_attn==2.7.4.post1
else
  echo "Activating virtual environment..."
  source "${venv_dir}/bin/activate" || exit 1
fi

exp_name="${exp_tag}-$(date +%Y%m%d-%H%M%S)"
output_dir="./exp/${exp_name}"
mkdir -p "${output_dir}"

echo "=================================================================="
echo "Starting Script at $(date)"
echo "Experiment Name: ${exp_name}"
echo "Output Directory: ${output_dir}"
echo "Python: $(which python)"
echo "=================================================================="

num_gpus="$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)"
if [ -z "$num_gpus" ] || ! [[ "$num_gpus" =~ ^[0-9]+$ ]]; then
    num_gpus=0
fi

if [ "${num_gpus}" -gt 1 ]; then
    accelerate_config_file="conf/multi_gpu.yaml"
    echo "Multiple GPUs detected (${num_gpus}). Using config: ${accelerate_config_file}"
elif [ "${num_gpus}" -eq 1 ]; then
    accelerate_config_file="conf/single_gpu.yaml"
    echo "Single GPU detected. Using config: ${accelerate_config_file}"
else
    echo "Error: No GPUs detected. This script requires at least one GPU."
    exit 1
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  echo ">>> STAGE 1: Starting Training <<<"
  cp "$0" "${output_dir}/run_script.sh" # Save the run script

  accelerate launch --config_file "${accelerate_config_file}" --num_processes ${num_gpus} train_sap.py \
    --model_name_or_path "microsoft/Phi-4-multimodal-instruct" \
    --data_dir "${DATASET_PATH}" \
    --tasks "${tasks[@]}" \
    --weights "${weights[@]}" \
    --aux_reg_cols "${aux_reg_cols[@]}" \
    --metric_for_best_model "${metric_for_best_model}" \
    --output_dir "${output_dir}" \
    --use_flash_attention \
    --batch_size_per_gpu ${batch_size_per_gpu} \
    --eval_batch_size_per_gpu ${eval_batch_size_per_gpu} \
    --test_batch_size_per_gpu ${test_batch_size_per_gpu} \
    --num_train_epochs ${epochs} \
    --learning_rate ${learning_rate} \
    --lr_scheduler "${lr_scheduler}" \
    --use_lora \
    --save_steps 500 \
    --eval_steps 500 \
    --min_duration ${min_duration} \
    --max_duration ${max_duration} \
    --use_aux_clf \
    --clf_loss_weight "${clf_loss_wgt}" \
    --aux_clf_use_enc_embeds \
    --use_aux_regression_task \
    --aux_reg_loss_weight "${clf_loss_wgt}" || exit 1
  echo ">>> STAGE 1 finished at $(date) <<<"
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
  echo ">>> STAGE 2: Starting Inference <<<"
  # Define output file paths
  hypo_file="${output_dir}/${inference_split}.hypo"
  clf_results_file="${output_dir}/classification_results_${inference_split}.json"

  python inference.py \
    --model_name_or_path "microsoft/Phi-4-multimodal-instruct" \
    --output_dir "${output_dir}" \
    --data_dir "${DATASET_PATH}" \
    --use_flash_attention \
    --test_batch_size_per_gpu ${test_batch_size_per_gpu} \
    --generation_max_length 448 \
    --tasks "${tasks[@]}" \
    --use_aux_clf \
    --classification_inference \
    --sapc_clf_output_name "${clf_results_file}" \
    --sapc_manifest_path "${SAPC_REF_PATH}" \
    --sapc_data_path "${SAPC_AUDIO_PATH}" \
    --sapc_output_name "${hypo_file}" \
    --sapc_split "${inference_split}" \
    $( [ "$apply_vad" = true ] && echo "--apply_vad" ) \
    $( [ "$cleanup_transcripts" = true ] && echo "--cleanup_transcripts" ) || exit 1
  echo ">>> STAGE 2 finished at $(date) <<<"
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
  echo ">>> STAGE 3: Starting SAPC Challenge Evaluation <<<"
  metrics_file="${output_dir}/sapc_metrics_${inference_split}.json"

  if [ -f "${output_dir}/${inference_split}.hypo" ]; then
    python sapc/utils/evaluate_local.py \
        --results_file "${metrics_file}" \
        --hypo_path "${output_dir}" \
        --ref_path "${SAPC_REF_PATH}"
  else
    echo "Hypothesis file not found in ${output_dir}. Skipping evaluation."
  fi
  echo ">>> STAGE 3 finished at $(date) <<<"
fi