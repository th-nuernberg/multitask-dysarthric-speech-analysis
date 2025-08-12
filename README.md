
# Joint ASR and Speech Attribute Prediction with Multimodal Language Models

This repository contains the official source code for the ASRU 2025 paper: 
*"Joint ASR and Speech Attribute Prediction for Conversational Dysarthric Speech Analysis with Multimodal Language Models"*.

This project introduces a multitask, conversational framework for analyzing dysarthric speech. 
The system is built upon [Phi-4-Multimodal](https://arxiv.org/abs/2503.01743) and is able to perform both Automatic Speech Recognition (ASR) 
and perceptual speech attribute rating simultaneously.

## Dataset

This project uses the **Speech Accessibility Project (SAP) dataset**. 
Please note that the dataset is **not included** in this repository. 
You have to acquire access to the dataset separately.

### Data Download and Preprocessing

1.  **Download Instructions:** You can apply for access to the SAP dataset via the official project page:
    - [https://speechaccessibilityproject.beckman.illinois.edu/conduct-research-through-the-project](https://speechaccessibilityproject.beckman.illinois.edu/conduct-research-through-the-project)

2.  **Data Preparation:** Once you have the data, you can use the official SAPC-template to preprocess the data:
    - [https://github.com/xiuwenz2/SAPC-template/tree/main](https://github.com/xiuwenz2/SAPC-template/tree/main)
    
    The output of this step will be a directory structure containing the raw `.wav` audio files and the corresponding manifest files (`.tsv`, `.wrd`, etc.) for each data split.

### Conversion to `datasets` Format

The training and inference scripts in this project read data using the Hugging Face `datasets` library, 
which requires the data to be compatible with the library. 

Note: This project does not provide the script for the data conversion. 
You are responsible for creating your own script to convert the `.wav` files and manifests into a `datasets`-compatible dataset. 

Your script should generate a dataset with the following structure, organized into `train/`, `val/`, and `test/` subdirectories:

-   **`utt_id`**: (string) The unique utterance identifier.
-   **`wav`**: (Audio) The audio data, which the `datasets` library will load from the `.wav` files.
-   **`words`**: (string) The reference transcription.
-   **`intelligibility`**, **`naturalness`**, etc.: (float/integer) Columns for each of the perceptual ratings.
-   Other columns as needed by your specific task configuration (see `constants.py` for a full list of columns the data loader can use).

After this step, your final data directory (e.g., `/path/to/your/prepared_sap_data`) should contain 
subdirectories for `train`, `val`, and `test`. 

## Installation

This project was developed and tested on Python 3.10. Newer versions (3.11+) are expected to work but have not been explicitly tested.

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:th-nuernberg/multitask-dysarthric-speech-analysis.git
    cd multitask-dysarthric-speech-analysis
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install required dependencies:**
    ```bash
    pip install --upgrade pip
    pip install wheel packaging
    pip install -r requirements.txt
    pip install flash_attn==2.7.4.post1
    ```

4.  **Install optional dependencies:**
    ```bash
    # For Voice Activity Detection (VAD) during inference (--apply_vad)
    pip install speechbrain==1.0.1
    # For logging experiments to Weights & Biases (--report_to wandb)
    pip install wandb
    ```

## Usage

The project is controlled via command-line arguments for training and inference. 
See [`cli_args.py`](cli_args.py) for a complete list of available options. 

For a full example showing model training with auxiliary classifiers and regression followed by inference and evaluation,
see the [example](examples/train_sap_multi_aux_clf_aux_reg.sh). 

### Important Prerequisite: Data Preparation

**Please ensure you have successfully prepared the SAP dataset before attempting to run any training or inference commands.**

Both `train_sap.py` and `inference.py` require the data to be processed into a format readable 
by the Hugging Face `datasets` library (e.g. parquet). 

For more details, refer to the **[Dataset](#dataset)** section above.

### Training

To finetune the model on the SAP dataset, use the `train_sap.py` script. 
You need to specify the data directory, output directory, and the tasks/weights for the multitask setup.

**Example Training Command:**
```bash
accelerate launch train_sap.py \
  --model_name_or_path "microsoft/Phi-4-multimodal-instruct" \
  --data_dir "/path/to/your/prepared_sap_data" \
  --output_dir "./experiments/phi4_finetuned_sap" \
  --tasks asr intelligibility naturalness consonants harsh monoloudness stress \
  --weights 0.14 0.14 0.14 0.14 0.14 0.14 0.14 \
  --use_flash_attention \
  --use_aux_clf \
  --unfreeze_audio_encoder \
  --batch_size_per_gpu 16 \
  --eval_batch_size_per_gpu 16 \
  --num_train_epochs 4 \
  --save_steps 500 \
  --eval_steps 500 \
  --learning_rate 4.0e-5 \
  --report_to "wandb"
```

### Inference and Evaluation

The inference and evaluation process is a two-step procedure. 

**Step 1: Run Inference**

The `inference.py` script is used to perform inference on a trained model. 
It has two functions when run with the `--classification_inference` flag:

1.  **ASR Hypothesis Generation**: It transcribes the audio files from a given split (e.g., `dev`) and saves the transcriptions to a `.hypo` file.
2.  **Classification Performance Evaluation**: It evaluates the model's ability to predict perceptual attributes (like intelligibility, naturalness, etc.) and saves the metrics to a `.json` file.

**Example Inference Command:**
```bash
python inference.py \
  --model_name_or_path "microsoft/Phi-4-multimodal-instruct" \
  --output_dir "./experiments/phi4_finetuned_sap" \
  --data_dir "/path/to/your/prepared_sap_data" \
  --test_batch_size_per_gpu 16 \
  --classification_inference \
  --sapc_output_name "./experiments/phi4_finetuned_sap/dev.hypo" \
  --sapc_clf_output_name "./experiments/phi4_finetuned_sap/classification_results.json" \
  --sapc_manifest_path "/path/to/sapc_template/manifests" \
  --sapc_data_path "/path/to/sapc_template/data/processed" \
  --sapc_split "dev" \
  --tasks asr intelligibility naturalness
```

**Step 2: Calculate SAP Challenge Metrics (WER and SemScore)**

After generating the hypothesis file (`.hypo`), use the `evaluate_local.py` script to compare it against the reference transcriptions. 
This script computes the official SAP Challenge metrics: Word Error Rate (WER) and SemScore.

**Example Evaluation Command:**
```bash
python evaluate_local.py \
  --hypo_path "./experiments/phi4_finetuned_sap/" \
  --ref_path "/path/to/sapc_template/manifests" \
  --results_file "./experiments/phi4_finetuned_sap/sapc_metrics.json"
```

This will generate a final JSON file containing the WER and SemScore for your model's ASR performance.


### Citation

```bibtex
@inproceedings{wagner2025joint,
  title={{Joint ASR and Speech Attribute Prediction for Conversational Dysarthric Speech Analysis with Multimodal Language Models}},
  author={Wagner, Dominik and Baumann, Ilja and Engert, Natalie and N{\"o}th, Elmar and Riedhammer, Korbinian and Bocklet, Tobias},
  booktitle={Proceedings of the 2025 IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU)},
  year={2025},
  organization={IEEE},
  note={To appear}
}
```
