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
"""
Finetune Phi-4-multimodal-instruct on the SAP dataset
Based on https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/sample_finetune_speech.py
"""

import functools
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
    Seq2SeqTrainingArguments,
    AutoConfig,
)
import logging
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from sklearn.metrics import f1_score

from cli_args import get_cli_args
from collator import collate_fn, custom_collate_fn
from constants import (
    CLF_SPECIAL_TOKEN,
    WER_METRIC,
    INSTRUCTION,
    ANSWER_SUFFIX,
    CLF_TASK_WEIGHTS,
    CLF_TASK2NUMCLASSES_MAP,
)
from custom_trainer import CustomSeq2SeqTrainer
from data import SampledSAPDataset, SAPTestDataset
from model.callback import (
    CosSimLoggingCallback,
    KLDLossLoggingCallback,
    AuxClfLossLoggingCallback,
    AuxRegLossLoggingCallback,
)
from model.custom_modeling_phi4mm import Phi4MMForCausalLM
from model.processing_phi4mm import (
    CustomPhi4MMProcessor,
    Phi4MMSpecAugmentAudioFeatureExtractor,
)
from util import (
    print_max_gpu_memory,
    print_parameter_count,
    check_for_clf_labels,
    compute_clf_metrics_by_task,
    compute_wer_metric_for_asr_tasks,
    print_weight_stats,
)

logger = logging.getLogger(__name__)


def compute_metrics_for_running_eval(pred, processor=None, task="asr"):
    decoded_labels = [
        processor.decode(
            _label_ids[(_label_ids > 0)], skip_special_tokens=True
        ).removesuffix(ANSWER_SUFFIX)
        for _label_ids in pred.label_ids
    ]

    decoded_preds = [
        processor.decode(
            _pred_ids[(_pred_ids > 0)], skip_special_tokens=True
        ).removesuffix(ANSWER_SUFFIX)
        for _pred_ids in pred.predictions
    ]

    # Remove the prompt from the output
    clean_decoded_preds = []
    for p in decoded_preds:
        for inst in INSTRUCTION.values():
            p = p.replace(inst, "")
        clean_decoded_preds.append(p)

    wer = 999
    try:
        logger.info(f"Computing WER based on {len(clean_decoded_preds)} predictions")
        wer = WER_METRIC.compute(
            predictions=clean_decoded_preds, references=decoded_labels
        )
    except Exception as e:
        logger.error(f"Unable to compute WER metric: {e}")
    for i in range(min(10, len(clean_decoded_preds))):
        logger.info(f"\n[{i}] hyp={clean_decoded_preds[i]} ||\nref={decoded_labels[i]}")

    _, clf_results = check_for_clf_labels(
        clean_decoded_preds, [task] * len(clean_decoded_preds)
    )
    _, clf_labels = check_for_clf_labels(decoded_labels, [task] * len(decoded_labels))
    filtered_labels = []
    filtered_predictions = []
    for label, pred in zip(clf_labels, clf_results):
        if label != -1:
            filtered_labels.append(label)
            filtered_predictions.append(pred)
    logger.info(
        f"Computing classification metrics for {len(filtered_labels)} "
        f"(total number of labels passed to eval function {len(decoded_labels)}) instances."
    )
    logger.info(f"GT labels: {Counter(filtered_labels)}")
    logger.info(f"Predicted labels: {Counter(filtered_predictions)}")
    if not filtered_labels:
        f1 = 0.0
    else:
        try:
            f1 = f1_score(filtered_labels, filtered_predictions, average="macro")
        except Exception as e:
            logger.error(f"Unable to compute F1 score {e}")
            f1 = -1
    final_results = {"wer": wer, "macro_f1": f1}
    logger.info(f"Results {final_results}")
    logger.info("=" * 80)
    return final_results


class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """Stopping criteria capable of receiving multiple stop-tokens and handling batched inputs."""

    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        """Initialize the multiple token batch stopping criteria.

        Args:
            stop_tokens: Stop-tokens.
            batch_size: Batch size.

        """

        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(
            batch_size, dtype=torch.long, device=stop_tokens.device
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # Only gather the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to `stop_tokens`
        generated_inputs = torch.eq(
            input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens
        )
        equal_generated_inputs = torch.all(generated_inputs, dim=2)

        # Mark the position where a stop token has been produced for each input in the batch,
        # but only if the corresponding entry is not already set
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]

        return torch.all(self.stop_tokens_idx)


def create_model(model_name_or_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        trust_remote_code=True,
    ).to("cuda")
    return model


def create_custom_model(cli_args):
    config = AutoConfig.from_pretrained(
        cli_args.model_name_or_path,
        trust_remote_code=True,
    )

    _clf_task_weights = None
    _clf_tasks = None
    if cli_args.use_aux_clf:
        _clf_task_weights = {}
        _clf_tasks = {}
        for task in cli_args.tasks:
            if "asr" not in task:
                _clf_task_weights[task] = CLF_TASK_WEIGHTS[task]
                _clf_tasks[task] = CLF_TASK2NUMCLASSES_MAP[task]

    config.clf_task_weights = _clf_task_weights  # Weighting of individual classification tasks Dict: {"taskname": "weight for loss"}
    config.clf_tasks = _clf_tasks  # Dict: {"taskname": num_classes}

    config.clf_loss_weight = (
        cli_args.clf_loss_weight
    )  # Overall weight of LLM loss vs. classifier loss
    config.use_gradient_consistency_loss = cli_args.use_gradient_consistency_loss
    config.use_kld_consistency_loss = cli_args.use_kld_consistency_loss
    config.kld_consistency_weight = cli_args.kld_consistency_weight
    config._attn_implementation = (
        "flash_attention_2" if cli_args.use_flash_attention else "sdpa"
    )

    if cli_args.aux_clf_use_enc_embeds:
        logger.info(
            f"{cli_args.aux_clf_use_enc_embeds=}: Setting config to output all hidden states."
        )

    config.output_hidden_states = cli_args.aux_clf_use_enc_embeds
    config.clf_use_enc_embeds = cli_args.aux_clf_use_enc_embeds

    if cli_args.apply_audio_encoder_lora:
        config.audio_encoder_lora = {
            "r": cli_args.lora_r,
            "lora_alpha": cli_args.lora_alpha,
            "layer": ["linear_q", "linear_v", "linear_out"],
            "dp": cli_args.lora_dropout,
        }
    # Aux regression task
    config.use_aux_regression_task = cli_args.use_aux_regression_task
    config.aux_reg_hidden_size = cli_args.aux_reg_hidden_size
    config.aux_reg_num_pred = cli_args.aux_reg_num_pred
    config.aux_reg_loss_weight = cli_args.aux_reg_loss_weight

    model = Phi4MMForCausalLM.from_pretrained(
        cli_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if cli_args.use_flash_attention else torch.float32,
        trust_remote_code=True,
    ).to("cuda")
    return model


def create_custom_processor(model_name_or_path, cr_ctc_adjustment=2.5):
    tmp_processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    custom_audio_feat_extractor = Phi4MMSpecAugmentAudioFeatureExtractor(
        audio_compression_rate=8,
        audio_downsample_rate=1,
        audio_feat_stride=1,
        cr_ctc_adjustment=cr_ctc_adjustment,
    )
    processor = CustomPhi4MMProcessor(
        tmp_processor.image_processor,
        custom_audio_feat_extractor,
        tmp_processor.tokenizer,
    )
    return processor


def aux_clf_forward_pass(mdl, inputs, tasks):
    # tasks: (batch_size, )
    # inputs.cls_labels: (batch_size, )
    # outputs.clf_logits: Dict: {task: tensor(batch_size, logits)}
    inputs.pop("labels", None)
    cls_labels = inputs.pop("cls_labels")  # LongTensor(batch,)

    outputs = mdl(
        **inputs, return_dict=True
    ).clf_logits  # Dict: {task: tensor(batch, logits)}

    lbls = []
    preds = []
    avail_tasks = []
    for i, task in enumerate(tasks):
        logits = outputs.get(task, None)
        if logits is not None:
            pred = int(torch.argmax(logits, dim=-1).cpu().numpy()[i])
            lbl = int(cls_labels[i].cpu().numpy())
            lbls.append(lbl)
            preds.append(pred)
            avail_tasks.append(task)
    return lbls, preds, avail_tasks


@torch.no_grad()
def run_eval(
    model,
    processor,
    eval_dataset,
    eval_collate_fn,
    save_path=None,
    disable_tqdm=False,
    eval_batch_size=1,
    eval_aux_clf=False,
):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    normalizer = BasicTextNormalizer()

    model.eval()
    all_generated_texts = []
    all_clf_results = []
    all_labels = []
    all_clf_labels = []
    all_tasks = []

    # Auxiliary classifier
    all_aux_clf_labels = []
    all_aux_clf_preds = []
    all_aux_clf_tasks = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=eval_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(
        stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt"
    )["input_ids"]
    stop_tokens_ids = stop_tokens_ids.to(f"cuda:{local_rank}")

    for inputs in tqdm(
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc="running eval"
    ):
        tasks = inputs.pop("tasks")
        stopping_criteria = StoppingCriteriaList(
            [
                MultipleTokenBatchStoppingCriteria(
                    stop_tokens_ids, batch_size=inputs.input_ids.size(0)
                )
            ]
        )
        inputs = inputs.to(f"cuda:{local_rank}")

        generated_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=128,
            stopping_criteria=stopping_criteria,
            num_logits_to_keep=0,  # Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all `input_ids`
        )

        stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(
            inputs.input_ids.size(0), -1
        )[:, 0]

        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )

        generated_text_with_special = [
            processor.decode(
                _pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).removesuffix(ANSWER_SUFFIX)
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]

        filtered_texts, clf_results = check_for_clf_labels(
            generated_text_with_special, tasks
        )

        all_generated_texts.extend(filtered_texts)
        all_clf_results.extend(clf_results)
        all_tasks.extend(tasks)

        labels = [
            processor.decode(_label_ids[_label_ids != 0]).removesuffix(ANSWER_SUFFIX)
            for _label_ids in inputs["labels"]
        ]
        labels, clf_labels = check_for_clf_labels(labels, tasks)

        if eval_aux_clf:
            aux_clf_lbls, aux_clf_pred, avail_tasks = aux_clf_forward_pass(
                model, inputs, tasks
            )
            all_aux_clf_labels.extend(aux_clf_lbls)
            all_aux_clf_preds.extend(aux_clf_pred)
            all_aux_clf_tasks.extend(avail_tasks)

        all_labels.extend(labels)
        all_clf_labels.extend(clf_labels)

    all_generated_texts = gather_object(all_generated_texts)
    all_clf_results = gather_object(all_clf_results)
    all_labels = gather_object(all_labels)
    all_clf_labels = gather_object(all_clf_labels)
    all_tasks = gather_object(all_tasks)

    if eval_aux_clf:
        all_aux_clf_labels = gather_object(all_aux_clf_labels)
        all_aux_clf_preds = gather_object(all_aux_clf_preds)
        all_aux_clf_tasks = gather_object(all_aux_clf_tasks)

    if rank == 0:
        assert len(all_generated_texts) == len(all_labels)
        logger.info(f"Computing WERs for {len(all_labels)} instances")
        wer_results = {"wer": -1}
        try:
            wer_results = compute_wer_metric_for_asr_tasks(
                WER_METRIC, normalizer, all_generated_texts, all_labels, all_tasks
            )
        except ValueError as e:
            logger.error(e)
        assert len(all_clf_results) == len(all_clf_labels)
        clf_results = compute_clf_metrics_by_task(
            all_clf_labels, all_clf_results, all_tasks
        )

        logger.info("Computed classification results:")
        for kk, vv in clf_results.items():
            logger.info(f"************ {kk} ************")
            for k, v in vv.items():
                logger.info(f"{k}: {v:.2f}")
            logger.info("=" * 80)

        # Auxiliary classifier metrics
        aux_clf_results = {}
        if eval_aux_clf:
            assert len(all_aux_clf_labels) == len(all_aux_clf_preds)
            aux_clf_results = compute_clf_metrics_by_task(
                all_aux_clf_labels,
                all_aux_clf_preds,
                all_aux_clf_tasks,
                name_prefix="aux_clf_",
            )
            logger.info("Computed auxiliary classifier results:")
            for kk, vv in aux_clf_results.items():
                logger.info(f"************ {kk} ************")
                for k, v in vv.items():
                    logger.info(f"{k}: {v:.2f}")
                logger.info("=" * 80)

        if save_path:
            logger.info(f"Saving results to {save_path}")
            with open(save_path, "w") as f:
                json.dump(
                    {"asr": wer_results} | clf_results | aux_clf_results, f, indent=2
                )

        return wer_results["wer"]
    return None


def main():
    formatter = "[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s >> %(message)s"
    logging.basicConfig(
        format=formatter,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    args = get_cli_args()
    accelerator = Accelerator()
    num_gpus = accelerator.num_processes

    if args.preprocessing_only:
        os.environ["WANDB_MODE"] = "disabled"
        logger.warning("--preprocessing_only is set. We won't log anything to wandb")
    if accelerator.is_main_process:
        logger.info(f"WANDB_MODE is set to: {os.environ.get('WANDB_MODE')}")

    with accelerator.local_main_process_first():
        if args.specaugment_mode is None and args.use_kld_consistency_loss:
            raise ValueError(
                f"--specaugment_mode can't be None when {args.use_kld_consistency_loss=}"
            )

        if args.specaugment_mode is not None:
            processor = create_custom_processor(
                args.model_name_or_path, cr_ctc_adjustment=args.cr_ctc_adjustment
            )
        else:
            processor = AutoProcessor.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True,
            )
        logger.info(
            f"SpecAugment mode is {args.specaugment_mode}. "
            f"Using {processor.__class__.__name__} with {processor.audio_processor.__class__.__name__}."
        )

        if (
            args.use_aux_clf
            or args.use_gradient_consistency_loss
            or args.use_aux_regression_task
            or args.specaugment_mode is not None
        ):
            model = create_custom_model(args)
        else:
            model = create_model(
                args.model_name_or_path,
                use_flash_attention=args.use_flash_attention,
            )
    if accelerator.is_main_process:
        logger.info(model.__class__.__name__)
        logger.info("Command line arguments:")
        logger.info(json.dumps(vars(args), indent=2))
    if args.use_lora:
        if accelerator.is_main_process:
            logger.info("Setting LoRA adapters in Phi4-Mini for speech component")

        adapters_to_set = ["speech"]
        if args.apply_audio_encoder_lora:
            if accelerator.is_main_process:
                logger.info("Setting LoRA adapters in audio encoder component")
            adapters_to_set.append("audio")
        model.set_lora_adapter(adapters_to_set)

    if args.unfreeze_audio_encoder:
        if accelerator.is_main_process:
            logger.info("Unfreezing audio encoder")
        with accelerator.local_main_process_first():

            def unfreeze_speech_components(model):
                """Directly unfreeze all audio components"""
                audio_embed = model.model.embed_tokens_extend.audio_embed
                audio_encoder = audio_embed.encoder
                audio_projection = audio_embed.audio_projection
                for component in [audio_embed, audio_encoder, audio_projection]:
                    for name, param in component.named_parameters():
                        param.requires_grad = True
                return model

            model = unfreeze_speech_components(model)

            encoder_params = list(
                model.model.embed_tokens_extend.audio_embed.encoder.parameters()
            )
            proj_params = list(
                model.model.embed_tokens_extend.audio_embed.audio_projection.parameters()
            )

            assert any(p.requires_grad for p in encoder_params), (
                "Speech encoder params frozen!"
            )
            assert any(p.requires_grad for p in proj_params), (
                "Speech output projection params frozen!"
            )
            if accelerator.is_main_process:
                logger.info(
                    "Parameter check passed. All audio encoder components were properly unfrozen."
                )

    # Add special token
    if args.add_special_clf_token:
        logger.info(f"Adding {CLF_SPECIAL_TOKEN} as a special token")
        special_tokens_dict = {"additional_special_tokens": [CLF_SPECIAL_TOKEN]}
        processor.tokenizer.add_special_tokens(special_tokens_dict)

        model.resize_token_embeddings(len(processor.tokenizer))
        model.get_input_embeddings().weight.requires_grad = True
        assert model.get_input_embeddings().weight.requires_grad

    if args.use_aux_clf and hasattr(model, "classification_heads"):
        logger.info("Unfreezing auxiliary classifiers")
        for head in model.classification_heads.values():
            for param in head.parameters():
                param.requires_grad = True

    if args.use_aux_regression_task and hasattr(model, "aux_reg"):
        logger.info("Unfreezing auxiliary regressor")
        for head in model.aux_reg:
            for param in head.parameters():
                param.requires_grad = True

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if accelerator.is_main_process:
        print_parameter_count(model)
        logger.info(f"{world_size=}")
    eval_dataset = SampledSAPDataset(
        processor,
        data_dir=args.data_dir,
        split=args.val_split_name,
        tasks=args.tasks,
        weights=args.weights,
        rank=rank,
        world_size=world_size,
        max_samples=args.max_eval_samples,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        use_aux_clf=args.use_aux_clf,
        aux_reg_cols=args.aux_reg_cols,
    )

    test_dataset = SAPTestDataset(
        processor,
        data_dir=args.data_dir,
        split="test",
        tasks=args.tasks,
        rank=rank,
        world_size=world_size,
        max_samples=args.max_test_samples,
        min_duration=0.1,
        max_duration=60,
        use_aux_clf=args.use_aux_clf,
    )

    train_dataset = SampledSAPDataset(
        processor,
        data_dir=args.data_dir,
        split="train",
        tasks=args.tasks,
        weights=args.weights,
        rank=rank,
        world_size=world_size,
        max_samples=args.max_train_samples,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        use_aux_clf=args.use_aux_clf,
        aux_reg_cols=args.aux_reg_cols,
    )

    if accelerator.is_main_process:
        logger.info(f"Training on {num_gpus} GPUs")
        logger.info(f"Number of samples in training data: {len(train_dataset)}")
        logger.info(f"Number of samples in validation data: {len(eval_dataset)}")
        logger.info(f"Number of samples in test data: {len(test_dataset)}")

    assert args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0, (
        "Batch size must be divisible by the number of GPUs"
    )
    gradient_accumulation_steps = args.batch_size // (
        num_gpus * args.batch_size_per_gpu
    )

    if args.use_flash_attention:
        fp16 = False
        bf16 = True
        fp16_full_eval = False
        bf16_full_eval = True
    else:
        fp16 = True
        bf16 = False
        fp16_full_eval = True
        bf16_full_eval = False

    if accelerator.is_main_process:
        logger.info(
            f"Using gradient checkpointing: {not args.disable_gradient_checkpointing}"
        )

    metric_for_best_model = args.metric_for_best_model
    if accelerator.is_main_process:
        logger.info(f"Using metric for best model: {metric_for_best_model}")

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.eval_batch_size_per_gpu,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=50,
        output_dir=args.output_dir,
        save_total_limit=3,
        save_only_model=True,  # Don't save optimizer states etc, so no picking up the training again
        bf16=bf16,
        fp16=fp16,
        fp16_full_eval=fp16_full_eval,
        bf16_full_eval=bf16_full_eval,
        remove_unused_columns=False,
        report_to=args.report_to,
        run_name=os.path.basename(args.output_dir),
        deepspeed=None,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,  # for unused SigLIP layers
        metric_for_best_model=metric_for_best_model,
        greater_is_better=metric_for_best_model != "wer",
        load_best_model_at_end=True,
        seed=424242,
        predict_with_generate=True,
        generation_max_length=128,  # Gets overridden by gen_kwargs anyway
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=50,
        dataloader_drop_last=args.dataloader_drop_last,
        ddp_timeout=int(os.environ.get("NCCL_TIMEOUT", 7200)),
    )

    # eval before fine-tuning
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not args.skip_first_eval:
        if accelerator.is_main_process:
            logger.info("Running eval before fine-tuning")
        score = run_eval(
            model,
            processor,
            test_dataset,
            custom_collate_fn
            if args.use_aux_clf
            or args.use_aux_regression_task
            or args.specaugment_mode is not None
            else collate_fn,
            save_path=out_path / "test_before.json",
            disable_tqdm=not args.tqdm,
            eval_batch_size=args.test_batch_size_per_gpu,
            eval_aux_clf=args.use_aux_clf,
        )
        if accelerator.is_main_process:
            logger.info(f"WER before fine-tuning: {score}")

    gen_kwargs = {
        "eos_token_id": processor.tokenizer.eos_token_id,
        "max_new_tokens": 128,
        "num_logits_to_keep": 0,
    }

    callbacks = []
    if args.use_gradient_consistency_loss:
        callbacks.append(CosSimLoggingCallback())
    if args.use_kld_consistency_loss:
        callbacks.append(KLDLossLoggingCallback())
    if args.use_aux_clf:
        callbacks.append(AuxClfLossLoggingCallback())
    if args.use_aux_regression_task:
        callbacks.append(AuxRegLossLoggingCallback())

    trainer = CustomSeq2SeqTrainer(
        model=model,
        gen_kwargs=gen_kwargs,
        args=training_args,
        data_collator=custom_collate_fn
        if args.use_aux_clf
        or args.use_aux_regression_task
        or args.specaugment_mode is not None
        else collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        compute_metrics=functools.partial(
            compute_metrics_for_running_eval, processor=processor
        ),
        callbacks=callbacks,
    )
    if accelerator.is_main_process:
        logger.info("Begin training.")
    trainer.train()
    logger.info(f"proc {accelerator.process_index} reached end of training")

    # Note: When load_best_model_at_end=True, this will save the best checkpoint instead of the last one.
    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")
        logger.info(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("All processes synchronized.")

    if accelerator.is_main_process:
        logger.info("Finished waiting for everyone. Attempting to clear GPU memory.")
        if args.use_aux_clf:
            logger.info("Weight statistics for auxiliary classifier before final eval:")
            print_weight_stats(model)
    # eval after fine-tuning (load saved checkpoint)
    # first try to clear GPU memory
    del model
    del trainer
    __import__("gc").collect()
    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        logger.info("Reloading model from disk for final eval.")
    # reload the model for final inference
    model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2"
        if args.use_flash_attention
        else "sdpa",
    ).to("cuda")

    score = run_eval(
        model,
        processor,
        test_dataset,
        custom_collate_fn
        if args.use_aux_clf
        or args.use_aux_regression_task
        or args.specaugment_mode is not None
        else collate_fn,
        save_path=out_path / "test_after.json",
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.test_batch_size_per_gpu,
        eval_aux_clf=args.use_aux_clf,
    )
    if accelerator.is_main_process:
        logger.info(f"WER after fine-tuning: {score}")
        print_max_gpu_memory()

        if args.use_aux_clf:
            logger.info("Weight statistics for auxiliary classifier after final eval:")
            print_weight_stats(model)
    accelerator.end_training()


if __name__ == "__main__":
    main()
