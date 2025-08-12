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
import json
import logging
import os
from pathlib import Path

import sys
import torch
from tqdm import tqdm
from transformers import (
    set_seed,
    AutoProcessor,
    AutoModelForCausalLM,
    StoppingCriteriaList,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from cli_args import get_cli_args
from collator import (
    sapc_inference_collate_fn,
    collate_fn,
    custom_collate_fn,
    pd_de_inference_collate_fn,
)
from constants import ANSWER_SUFFIX, WER_METRIC, CLF_SPECIAL_TOKEN
from data import SAPTSVInferenceDataset, SAPTestDataset, PDGermanInferenceDataset
from train_sap import (
    MultipleTokenBatchStoppingCriteria,
    aux_clf_forward_pass,
    create_custom_processor,
)
from util import (
    clean_repetitions,
    print_max_gpu_memory,
    check_for_clf_labels,
    compute_wer_metric_for_asr_tasks,
    compute_clf_metrics_by_task,
    print_weight_stats,
    write_preds_to_csv,
)
from vad import load_vad_model

logger = logging.getLogger(__name__)


@torch.no_grad()
def sapc_inference_loop(
    model,
    processor,
    eval_dataset,
    disable_tqdm=False,
    eval_batch_size=1,
    max_new_tokens=64,
):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model.eval()
    all_generated_texts = []
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=sapc_inference_collate_fn,
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
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc="ASR eval"
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
            max_new_tokens=max_new_tokens,
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
        filtered_texts, _ = check_for_clf_labels(generated_text_with_special, tasks)
        all_generated_texts.extend(filtered_texts)

    return all_generated_texts


@torch.no_grad()
def pd_de_inference_loop(
    model,
    processor,
    eval_dataset,
    disable_tqdm=False,
    eval_batch_size=1,
    max_new_tokens=64,
):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model.eval()
    all_generated_texts = []
    all_labels = []
    all_utt_ids = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=pd_de_inference_collate_fn,
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
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc="PD_DE ASR eval"
    ):
        tasks = inputs.pop("tasks")
        labels = inputs.pop("labels")
        utt_ids = inputs.pop("utt_ids")
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
            max_new_tokens=max_new_tokens,
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
        filtered_texts, _ = check_for_clf_labels(generated_text_with_special, tasks)
        all_generated_texts.extend(filtered_texts)

        labels = [
            processor.decode(_label_ids[_label_ids != 0]).removesuffix(ANSWER_SUFFIX)
            for _label_ids in labels
        ]
        labels, _ = check_for_clf_labels(labels, tasks)
        all_labels.extend(labels)
        all_utt_ids.extend(utt_ids)

    return all_generated_texts, all_labels, all_utt_ids


@torch.no_grad()
def classification_inference_loop(
    model,
    processor,
    eval_dataset,
    save_path=None,
    disable_tqdm=False,
    max_new_tokens=64,
    eval_batch_size=1,
    eval_aux_clf=False,
    specaugment_mode=None,
    zero_shot_inference=False,
):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    normalizer = BasicTextNormalizer()

    model.eval()

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=custom_collate_fn
        if eval_aux_clf or specaugment_mode is not None
        else collate_fn,
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

    all_generated_texts = []
    all_clf_results = []
    all_labels = []
    all_clf_labels = []
    all_tasks = []

    # Auxiliary classifier
    all_aux_clf_labels = []
    all_aux_clf_preds = []
    all_aux_clf_tasks = []

    for inputs in tqdm(
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc="Classifier eval"
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
            max_new_tokens=max_new_tokens,
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
            generated_text_with_special, tasks, is_zero_shot=zero_shot_inference
        )
        all_generated_texts.extend(filtered_texts)
        all_clf_results.extend(clf_results)
        all_tasks.extend(tasks)

        labels = [
            processor.decode(_label_ids[_label_ids != 0]).removesuffix(ANSWER_SUFFIX)
            for _label_ids in inputs["labels"]
        ]
        labels, clf_labels = check_for_clf_labels(labels, tasks)
        all_labels.extend(labels)
        all_clf_labels.extend(clf_labels)

        if eval_aux_clf:
            aux_clf_lbls, aux_clf_pred, avail_tasks = aux_clf_forward_pass(
                model, inputs, tasks
            )
            all_aux_clf_labels.extend(aux_clf_lbls)
            all_aux_clf_preds.extend(aux_clf_pred)
            all_aux_clf_tasks.extend(avail_tasks)

    if rank == 0:
        assert len(all_generated_texts) == len(all_labels)
        logger.info(f"Computing WERs for {len(all_labels)} instances")
        wer_results = {"wer": -1}
        try:
            # results = compute_wer_metric(WER_METRIC, normalizer, all_generated_texts, all_labels)
            wer_results = compute_wer_metric_for_asr_tasks(
                WER_METRIC, normalizer, all_generated_texts, all_labels, all_tasks
            )
        except ValueError as e:
            logger.error(e)
        assert len(all_clf_results) == len(all_clf_labels)
        # clf_results = compute_clf_metrics(all_clf_labels, all_clf_results)
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
            with open(save_path, "w") as f:
                json.dump(
                    {"asr": wer_results} | clf_results | aux_clf_results, f, indent=2
                )


def run_sapc_eval(inference_args):
    # inference_args = get_cli_args()
    set_seed(424242)
    processor = AutoProcessor.from_pretrained(
        inference_args.model_name_or_path,
        trust_remote_code=True,
    )
    # Add special token
    if inference_args.add_special_clf_token:
        logger.info(f"Adding {CLF_SPECIAL_TOKEN} as a special token")
        special_tokens_dict = {"additional_special_tokens": [CLF_SPECIAL_TOKEN]}
        processor.tokenizer.add_special_tokens(special_tokens_dict)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    vad_model = None
    if inference_args.apply_vad:
        logger.info("Using VAD in preprocessing")
        vad_model = load_vad_model(inference_args.sapc_root)

    test_dataset = SAPTSVInferenceDataset(
        processor,
        inference_args,
        rank=rank,
        world_size=world_size,
        min_duration=0.01,
        max_duration=500,
        vad_model=vad_model,
        zero_shot_inference=inference_args.zero_shot_inference,
    )

    if inference_args.apply_vad:
        del vad_model
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        inference_args.output_dir,
        torch_dtype=torch.bfloat16
        if inference_args.use_flash_attention
        else torch.float32,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2"
        if inference_args.use_flash_attention
        else "sdpa",
    ).to("cuda")

    results = sapc_inference_loop(
        model,
        processor,
        test_dataset,
        disable_tqdm=not inference_args.tqdm,
        eval_batch_size=inference_args.test_batch_size_per_gpu,
        max_new_tokens=inference_args.generation_max_length,
    )

    if inference_args.cleanup_transcripts:
        results = [
            clean_repetitions(r, ngram_range=(1, 3), repetition_threshold=5)
            for r in results
        ]

    logger.info(
        f"Writing transcripts to {inference_args.sapc_output_name} "
        f"(number of transcriptions: {len(results)})..."
    )
    Path(inference_args.sapc_output_name).parent.mkdir(exist_ok=True, parents=True)
    with open(inference_args.sapc_output_name, "w") as output_file:
        for r in results:
            output_file.write(r + "\n")

    logger.info("SAPC evaluation complete.")
    print_max_gpu_memory()


def run_pd_de_eval(inference_args):
    set_seed(424242)
    processor = AutoProcessor.from_pretrained(
        inference_args.model_name_or_path,
        trust_remote_code=True,
    )
    # Add special token
    if inference_args.add_special_clf_token:
        logger.info(f"Adding {CLF_SPECIAL_TOKEN} as a special token")
        special_tokens_dict = {"additional_special_tokens": [CLF_SPECIAL_TOKEN]}
        processor.tokenizer.add_special_tokens(special_tokens_dict)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    vad_model = None
    if inference_args.apply_vad:
        logger.info("Using VAD in preprocessing")
        vad_model = load_vad_model(inference_args.sapc_root)

    test_dataset = PDGermanInferenceDataset(
        processor,
        inference_args,
        split="train",
        rank=rank,
        world_size=world_size,
        min_duration=0.01,
        max_duration=500,
        vad_model=vad_model,
        zero_shot_inference=inference_args.zero_shot_inference,
    )

    if inference_args.apply_vad:
        del vad_model
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        inference_args.output_dir,
        torch_dtype=torch.bfloat16
        if inference_args.use_flash_attention
        else torch.float32,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2"
        if inference_args.use_flash_attention
        else "sdpa",
    ).to("cuda")

    results, labels, utt_ids = pd_de_inference_loop(
        model,
        processor,
        test_dataset,
        disable_tqdm=not inference_args.tqdm,
        eval_batch_size=inference_args.test_batch_size_per_gpu,
        max_new_tokens=inference_args.generation_max_length,
    )

    if inference_args.cleanup_transcripts:
        results = [
            clean_repetitions(r, ngram_range=(1, 3), repetition_threshold=5)
            for r in results
        ]

    Path(inference_args.sapc_output_name).parent.mkdir(exist_ok=True, parents=True)
    # csv_filename = os.path.join(inference_args.sapc_output_name, "test_predictions.csv")
    write_preds_to_csv((utt_ids, results, labels), inference_args.sapc_output_name)
    logger.info(f"Wrote predictions to {inference_args.sapc_output_name}")

    logger.info("PD_DE evaluation complete.")
    print_max_gpu_memory()


def run_classification_eval(inference_args):
    set_seed(424242)
    if inference_args.specaugment_mode is not None:
        processor = create_custom_processor(
            inference_args.model_name_or_path,
            cr_ctc_adjustment=inference_args.cr_ctc_adjustment,
        )
    else:
        processor = AutoProcessor.from_pretrained(
            inference_args.model_name_or_path,
            trust_remote_code=True,
        )
    # Add special token
    if inference_args.add_special_clf_token:
        logger.info(f"Adding {CLF_SPECIAL_TOKEN} as a special token")
        special_tokens_dict = {"additional_special_tokens": [CLF_SPECIAL_TOKEN]}
        processor.tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(
        f"SpecAugment mode is {inference_args.specaugment_mode}. Using {processor.__class__.__name__} "
        f"with {processor.audio_processor.__class__.__name__}."
    )
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    test_dataset = SAPTestDataset(
        processor,
        data_dir=inference_args.data_dir,
        split="test",
        tasks=inference_args.tasks,
        rank=rank,
        world_size=world_size,
        max_samples=inference_args.max_test_samples,
        min_duration=0.01,
        max_duration=500,
        use_aux_clf=inference_args.use_aux_clf,
        zero_shot_inference=inference_args.zero_shot_inference,
    )

    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        inference_args.output_dir,
        torch_dtype=torch.bfloat16
        if inference_args.use_flash_attention
        else torch.float32,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2"
        if inference_args.use_flash_attention
        else "sdpa",
    ).to("cuda")

    logger.info(model.config)

    if inference_args.use_aux_clf:
        logger.info("Weight statistics for auxiliary classifier:")
        print_weight_stats(model)

    Path(inference_args.sapc_clf_output_name).parent.mkdir(exist_ok=True, parents=True)
    classification_inference_loop(
        model,
        processor,
        test_dataset,
        disable_tqdm=not inference_args.tqdm,
        eval_batch_size=inference_args.test_batch_size_per_gpu,
        max_new_tokens=inference_args.generation_max_length,
        save_path=inference_args.sapc_clf_output_name,
        eval_aux_clf=inference_args.use_aux_clf,
        specaugment_mode=inference_args.specaugment_mode,
        zero_shot_inference=inference_args.zero_shot_inference,
    )

    logger.info("SAPC classifier evaluation complete.")
    print_max_gpu_memory()


def main():
    inference_args = get_cli_args()
    if inference_args.classification_inference:
        logger.info("Starting classification inference")
        run_classification_eval(inference_args)

    if inference_args.pd_de_inference:
        logger.info("Starting PD_DE ASR eval")
        run_pd_de_eval(inference_args)
    else:
        logger.info("Starting SAPC ASR eval")
        run_sapc_eval(inference_args)


if __name__ == "__main__":
    formatter = "[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s >> %(message)s"
    logging.basicConfig(
        format=formatter,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    main()
