# Copyright 2025 Technische Hochschule Nürnberg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Author: Dominik Wagner, Technische Hochschule Nürnberg
import argparse


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-multimodal-instruct",
        help="Model name or path to load from",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/nfs/scratch/staff/wagnerdo/data/speech_accessibility/parquet/whsp_spk_adapt_official_preprocessing",
        help="Parquet dataset",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="wer",
        help="Either 'wer' or 'macro_f1' to determine best model",
    )
    parser.add_argument(
        "--tasks", nargs="+", type=str, required=False, help="List of task names"
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        required=False,
        help="List of weights for tasks",
    )
    parser.add_argument(
        "--val_split_name",
        type=str,
        default="val",
        help="Directory name of the validation split.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        help="Learning rate scheduler (e.g. 'linear' or 'cosine')",
    )
    parser.add_argument(
        "--use_flash_attention", action="store_true", help="Use Flash Attention"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Continue training LoRA adapters for the speech component in the LLM decoder.",
    )
    parser.add_argument(
        "--unfreeze_audio_encoder",
        action="store_true",
        help="Make conformer audio encoder trainable.",
    )
    parser.add_argument(
        "--skip_first_eval",
        action="store_true",
        help="Skip evaluation before fine-tuning.",
    )
    parser.add_argument(
        "--disable_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--dataloader_drop_last", action="store_true", help="Drop last incomplete batch"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./exp", help="Output directory"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--batch_size_per_gpu", type=int, default=32, help="Batch size per GPU"
    )
    parser.add_argument(
        "--eval_batch_size_per_gpu",
        type=int,
        default=32,
        help="Batch size per GPU for continuous eval in trainer",
    )
    parser.add_argument(
        "--test_batch_size_per_gpu",
        type=int,
        default=32,
        help="Batch size per GPU for test set",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Max number of test samples to use from dataset",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Max number of samples for continuous eval during training",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Max number of train samples to use from dataset",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Max number of train samples to use from dataset",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Max number of train samples to use from dataset",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=4.0e-5, help="Learning rate"
    )
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.0,
        help="Minimum audio duration for training and eval",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Maximum audio duration for training and eval",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Weight decay")
    parser.add_argument(
        "--no-tqdm", dest="tqdm", action="store_false", help="Disable tqdm"
    )
    parser.add_argument(
        "--preprocessing_only",
        action="store_true",
        help="Only preprocess data; no training/eval",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Set integration to report the results and logs to.",
    )
    # SAPC/classification inference args
    parser.add_argument(
        "--classification_inference",
        action="store_true",
        help="Only preprocess data; no training/eval",
    )
    parser.add_argument(
        "--sapc_manifest_path",
        type=str,
        default=None,
        help="SAPC Path to the .tsv file e.g. /taiga/manifest.",
    )
    parser.add_argument(
        "--sapc_data_path",
        type=str,
        default=None,
        help="SAPC Base directory containing audio files: /taiga/data/processed",
    )
    parser.add_argument(
        "--sapc_output_name",
        type=str,
        default=None,
        help="SAPC Output file name for predictions: /taiga/downloads/???/???/inference/???.hypo",
    )
    parser.add_argument(
        "--sapc_clf_output_name",
        type=str,
        default=None,
        help="SAPC Path to the .tsv file e.g.  /taiga/downloads/???/???/inference/???.json",
    )
    parser.add_argument(
        "--sapc_root",
        type=str,
        default=None,
        help="SAPC root directory: /taiga/downloads/???/???",
    )
    parser.add_argument(
        "--sapc_split",
        type=str,
        default=None,
        help="SAPC test split. Used to find manifest file and audio folder.",
    )
    parser.add_argument(
        "--apply_vad",
        action="store_true",
        help="Whether to apply VAD as a preprocessing step.",
    )
    parser.add_argument(
        "--cleanup_transcripts",
        action="store_true",
        help="Clean repeated n-grams after inference.",
    )
    parser.add_argument(
        "--generation_max_length",
        type=int,
        default=64,
        help="Max number of tokens to generate",
    )

    # Aux classifier args
    parser.add_argument(
        "--clf_loss_weight",
        type=float,
        default=1.0,
        help="Overall weight of LLM loss vs. classifier loss. When --use_gradient_consistency_loss it performs the weighting on cosine similarity part. ",
    )
    parser.add_argument(
        "--use_aux_clf",
        action="store_true",
        help="Employ auxiliary classifiers with linear heads in addition to token-based classification",
    )
    parser.add_argument(
        "--add_special_clf_token",
        action="store_true",
        help="Add a special classification token; requires training the embedding table (~600M params)",
    )
    parser.add_argument(
        "--aux_clf_use_enc_embeds",
        action="store_true",
        help="Use audio encoder embeddings instead of decoder hidden states in auxiliary classifier",
    )

    # Aux Regression args
    parser.add_argument(
        "--aux_reg_cols",
        nargs="+",
        type=str,
        required=False,
        default=None,
        help="Column names considered during auxiliary regression",
    )
    parser.add_argument(
        "--use_aux_regression_task",
        action="store_true",
        help="Use pretrained embeddings in an auxiliary regression task.",
    )
    parser.add_argument("--aux_reg_hidden_size", type=int, default=384)
    parser.add_argument("--aux_reg_num_pred", type=int, default=5)
    parser.add_argument("--aux_reg_loss_weight", type=float, default=1.0)

    # SpecAug args
    parser.add_argument(
        "--specaugment_mode",
        type=str,
        default=None,
        help="Mode for contrastive SpecAugment (None means no SpecAugment is applied). Modes: "
        "(1) clean_vs_aug: Returns a clean and an augmented spectrogram. "
        "(2) aug_vs_aug: Returns two augmented spectrograms of same length but different time and frequency maskings "
        "(3) aug_only: Returns the same augmented spectrogram twice "
        "(4) clean_only: Returns the same non-augmented spectrogram twice",
    )
    parser.add_argument(
        "--cr_ctc_adjustment",
        type=float,
        default=2.5,
        help="The default SpecAugment settings are according to "
        "Lhotse (https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/signal_transforms.py). "
        "In 'CR-CTC: Consistency regularization on CTC for improved speech recognition' (https://arxiv.org/abs/2410.05101) "
        "both the number of time masking regions and the maximum masking fraction is increased by a factor of 2.5.",
    )
    # Consistency loss args
    parser.add_argument(
        "--use_gradient_consistency_loss",
        action="store_true",
        help="Encourage alignment between LM and CLF loss by adding a gradient-based consistency term $\mathcal{L}_{\text{total}} = \mathcal{L}_1 + \mathcal{L}_2 - \lambda \cdot \cos(\nabla_\theta \mathcal{L}_1, \nabla_\theta \mathcal{L}_2)$",
    )
    parser.add_argument(
        "--use_kld_consistency_loss",
        action="store_true",
        help="Encourage alignment between two differently augmented spectrograms by adding a KLD-based consistency term",
    )
    parser.add_argument(
        "--kld_consistency_weight",
        type=float,
        default=0.2,
        help="Weight of aug consistency loss vs. regular loss. This is equivalent to alpha in the CR-CTC paper. They use alphas [0.1, 0.2, 0.3]",
    )

    #################### Audio Encoder LoRA Args ##################################
    parser.add_argument("--apply_audio_encoder_lora", action="store_true")
    # parser.add_argument("--use_dora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.01)
    # parser.add_argument("--lora_target_modules", type=str, default="linear_q|linear_v|linear_out")

    #################### Zero-Shot Inference ##################################
    parser.add_argument("--zero_shot_inference", action="store_true")

    #################### Inference on German PD data ##################################
    parser.add_argument("--pd_de_inference", action="store_true")
    return parser.parse_args()
