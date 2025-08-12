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
import functools
import math
import os
import random
from pathlib import Path

import datasets
import torch
import logging

from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import (
    ANSWER_SUFFIX,
    INSTRUCTION,
    CLF_SPECIAL_TOKEN,
    _IGNORE_INDEX,
    COLUMNS_TO_KEEP,
    TASK2CLFCOL_MAP,
    ETIOLOGY_MAP,
    CATEGORY_MAP,
    ZERO_SHOT_INSTRUCTION,
)
from model.processing_phi4mm import Phi4MMSpecAugmentAudioFeatureExtractor
from vad import run_vad

logger = logging.getLogger(__name__)


def get_clf_string(data, task, is_multi=False):
    rating_str = ""
    clf_str = CLF_SPECIAL_TOKEN
    if "intelligibility" in task:
        score1 = data.get("intelligbility", "").strip()
        score2 = data.get("intelligibility", "").strip()
        if is_multi:
            clf_str = "<|intell|>"
        if len(score1) > 0:
            rating_str += f"{clf_str}{score1}"
        elif len(score2) > 0:
            rating_str += f"{clf_str}{score2}"
    if "naturalness" in task:
        if is_multi:
            clf_str = "<|nat|>"
        score = data.get("naturalness", "").strip()
        if len(score) > 0:
            rating_str += f"{clf_str}{score}"
    if "etiology" in task:
        if is_multi:
            clf_str = "<|etio|>"
        score = data.get("etiology", "").strip()
        if len(score) > 0:
            rating_str += f"{clf_str}{score}"
    if "category" in task:
        if is_multi:
            clf_str = "<|cat|>"
        score = data.get("category", "").strip()
        if len(score) > 0:
            rating_str += f"{clf_str}{score}"
    if "harsh" in task:
        if is_multi:
            clf_str = "<|harsh|>"
        score = data.get("harsh_voice", "").strip()
        if len(score) > 0:
            rating_str += f"{clf_str}{score}"
    if "consonants" in task:
        if is_multi:
            clf_str = "<|cons|>"
        score = data.get("imprecise_consonants", "").strip()
        if len(score) > 0:
            rating_str += f"{clf_str}{score}"
    if "stress" in task:
        if is_multi:
            clf_str = "<|stress|>"
        score = data.get("reduced_stress", "").strip()
        if len(score) > 0:
            rating_str += f"{clf_str}{score}"
    if "monoloudness" in task:
        if is_multi:
            clf_str = "<|mono|>"
        score = data.get("monoloudness", "").strip()
        if len(score) > 0:
            rating_str += f"{clf_str}{score}"
    if task == "asr":
        # if is_aux_clf:
        #    rating_str += f"{clf_str}1"
        # else:
        rating_str = ""
    return rating_str


def load_sapc_tsv_test_set(inference_args, vad_model=None):
    logger.info("Processing SAPC TSV file...")
    data = []
    manifest = os.path.join(
        inference_args.sapc_manifest_path, f"{inference_args.sapc_split}.tsv"
    )
    with open(manifest, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        for line in tqdm(lines, desc="Data loading"):
            audio_filename = line.strip().split()[0].split("/")[-1]
            full_path = os.path.join(
                inference_args.sapc_data_path, inference_args.sapc_split, audio_filename
            )
            if not os.path.exists(full_path):
                logger.warning(f"Audio file {full_path} does not exist. Skipping.")
                continue

            if inference_args.apply_vad and vad_model is not None:
                retries = 0
                while retries < 3:
                    try:
                        audio_input, num_samples = run_vad(full_path, vad_model)
                        retries = 99
                    except Exception as e:
                        logger.error(f"Unable to run VAD: {e} (retry {retries})")
                        retries += 1
                        num_samples = 0
                if num_samples < 12_000:
                    logger.warning(
                        f"Input after VAD is too short ({num_samples} samples). Falling back to full waveform!"
                    )
                    with open(full_path, "rb") as af:
                        audio_input = af.read()
            else:
                with open(full_path, "rb") as af:
                    audio_input = af.read()
            data.append({"wav": audio_input})
    logger.info("Creating HF Dataset...")
    return datasets.Dataset.from_list(data).with_format("numpy")


class SAPDataset(Dataset):
    def __init__(
        self,
        processor,
        data_dir,
        split,
        task="asr",
        rank=0,
        world_size=1,
        max_samples=None,
        min_duration=1.0,
        max_duration=30.0,
    ):
        files = list((Path(data_dir) / split).rglob("*.parquet"))
        if rank == 0:
            logger.info(
                f"Loading data for split {split}. Found {len(files)} parquet files."
            )
        self.data = load_dataset(
            data_dir, data_files={f"{split}": [str(f) for f in files]}
        )[split]
        if max_samples is not None:
            self.data = self.data.shuffle(seed=424242)
            self.data = self.data.select(range(max_samples))

        self.data = self.data.remove_columns(
            [col for col in self.data.column_names if col not in COLUMNS_TO_KEEP]
        )
        self.data = self.data.cast_column(
            "wav", datasets.features.Audio(sampling_rate=16_000)
        )
        self.data = self.data.filter(
            lambda x: min_duration
            <= len(x["wav"]["array"]) / x["wav"]["sampling_rate"]
            <= max_duration
        )
        if rank == 0:
            logger.info("Data after filtering:")
            logger.info(self.data)
        self.training = "train" in split or "val" in split
        self.processor = processor
        self.task = task
        self.instruction = INSTRUCTION[task]

        if world_size > 1:
            self.data = self.data.shard(world_size, rank)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Required fields: "wav", "utt_id", "intelligbility", "intelligibility", "naturalness", "words", "etiology", "category"
        This will use answer = f"{transcript}{clf_str}{ANSWER_SUFFIX}" as labels.
        This means we will have <|clf|> followed by the classifier label in the label ids.
        """
        data = self.data[idx]
        clf_str = get_clf_string(data, self.task)
        # No rating available for this example, so we fall back to just doing ASR
        needs_fallback_instruction = len(clf_str) < 1
        if needs_fallback_instruction:
            self.instruction = INSTRUCTION["asr"]
            self.task = "asr"

        user_message = {
            "role": "user",
            "content": "<|audio_1|>\n" + self.instruction,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            audios=[(data["wav"]["array"], data["wav"]["sampling_rate"])],
            return_tensors="pt",
        )
        transcript = data["words"] if "asr" in self.task else ""
        answer = f"{transcript}{clf_str}{ANSWER_SUFFIX}"

        answer_ids = self.processor.tokenizer(answer, return_tensors="pt").input_ids
        task_dict = {}
        if self.training:
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1] :] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids
            task_dict = {"tasks": self.task}

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
            **task_dict,
        }


class SampledSAPDataset(Dataset):
    def __init__(
        self,
        processor,
        data_dir,
        split,
        tasks,
        weights,
        rank=0,
        world_size=1,
        max_samples=None,
        min_duration=1.0,
        max_duration=30.0,
        use_aux_clf=False,
        specaugment_mode="aug_vs_aug",
        aux_reg_cols=None,
    ):
        assert len(tasks) == len(weights), "Length of tasks and weights must match"
        if aux_reg_cols is not None and len(aux_reg_cols) < 1:
            aux_reg_cols = None
        self.tasks = tasks
        self.weights = weights
        self.use_aux_clf = use_aux_clf
        self.specaugment_mode = specaugment_mode
        self.aux_reg_cols = aux_reg_cols

        files = list((Path(data_dir) / split).rglob("*.parquet"))
        if rank == 0:
            logger.info(
                f"[SampledSAPDataset] Loading data for split {split}. Found {len(files)} parquet files."
            )
        self.data = load_dataset(
            data_dir, data_files={f"{split}": [str(f) for f in files]}
        )[split]
        self.data = self.data.shuffle(seed=424242)
        if max_samples is not None:
            self.data = self.data.select(range(max_samples))

        self.data = self.data.remove_columns(
            [col for col in self.data.column_names if col not in COLUMNS_TO_KEEP]
        )
        self.data = self.data.cast_column(
            "wav", datasets.features.Audio(sampling_rate=16_000)
        )
        self.data = self.data.filter(
            lambda x: min_duration
            <= len(x["wav"]["array"]) / x["wav"]["sampling_rate"]
            <= max_duration
        )
        if rank == 0:
            logger.info("[SampledSAPDataset] Data after filtering:")
            logger.info(self.data)
        self.training = "train" in split
        self.validation = "val" in split
        self.processor = processor

        if world_size > 1:
            self.data = self.data.shard(world_size, rank)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        task = random.choices(self.tasks, weights=self.weights, k=1)[0]
        instruction = INSTRUCTION[task]
        clf_str = get_clf_string(data, task)
        needs_fallback_instruction = len(clf_str) < 1
        if needs_fallback_instruction:
            instruction = INSTRUCTION["asr"]
            task = "asr"
            clf_str = get_clf_string(data, task)

        # Extract cls_labels for aux classifier
        # Set dummy score for ASR task
        clf_score = 1
        if self.use_aux_clf and task != "asr":
            try:
                candidate = clf_str.replace(CLF_SPECIAL_TOKEN, "").strip()
                combined_mappings = ETIOLOGY_MAP | CATEGORY_MAP
                for k, v in combined_mappings.items():
                    if candidate == k:
                        candidate = v + 1
                        break
                clf_score = int(candidate) - 1
            except Exception as e:
                logger.error(
                    f"Unable to obtain integer value from classification string '{clf_str}'. "
                    f"Falling back to default value. {e}"
                )

        user_message = {
            "role": "user",
            "content": "<|audio_1|>\n" + instruction,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )

        additional_kwargs = {}
        if isinstance(
            self.processor.audio_processor, Phi4MMSpecAugmentAudioFeatureExtractor
        ):
            additional_kwargs["specaugment_mode"] = self.specaugment_mode
        inputs = self.processor(
            text=prompt,
            audios=[(data["wav"]["array"], data["wav"]["sampling_rate"])],
            return_tensors="pt",
            **additional_kwargs,
        )
        spec_aug_dict = {}
        if hasattr(inputs, "input_audio_embeds_other"):
            spec_aug_dict["input_audio_embeds_other"] = inputs.input_audio_embeds_other

        transcript = data["words"] if "asr" in task else ""
        answer = f"{transcript}{clf_str}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors="pt").input_ids

        task_dict = {}
        aux_clf_dict = {}
        if self.training:
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1] :] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids

        if self.use_aux_clf:
            aux_clf_dict["cls_labels"] = clf_score
            task_dict = {"tasks": task}

        aux_reg_dict = {}
        if self.aux_reg_cols is not None:
            if "reg_embed" in data and data["reg_embed"] is not None:
                try:
                    reg_embed_tensor = torch.tensor(
                        data["reg_embed"], dtype=torch.float
                    )
                    aux_reg_dict["aux_reg_embeds"] = reg_embed_tensor
                    # logger.error(f"{reg_embed_tensor.shape=}")
                except Exception as e:
                    logger.error(f"Failed to convert 'reg_embed' to tensor: {e}")

            values = [data.get(col, None) for col in self.aux_reg_cols]
            try:
                valid = all(
                    v is not None
                    and not (isinstance(v, str) and v.strip() == "")
                    and not (isinstance(v, float) and math.isnan(v))
                    for v in values
                )
                if valid:
                    aux_reg_dict["aux_reg_labels"] = torch.tensor(
                        [float(v) for v in values], dtype=torch.float
                    )
                    aux_reg_dict["aux_reg_mask"] = torch.ones(
                        len(values), dtype=torch.bool
                    )
                else:
                    aux_reg_dict["aux_reg_labels"] = torch.zeros(
                        len(values), dtype=torch.float
                    )
                    aux_reg_dict["aux_reg_mask"] = torch.zeros(
                        len(values), dtype=torch.bool
                    )
            except Exception:
                aux_reg_dict["aux_reg_labels"] = torch.zeros(
                    len(values), dtype=torch.float
                )
                aux_reg_dict["aux_reg_mask"] = torch.zeros(
                    len(values), dtype=torch.bool
                )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
            **task_dict,
            **aux_clf_dict,
            **spec_aug_dict,
            **aux_reg_dict,
        }


class SAPTestDataset(Dataset):
    def __init__(
        self,
        processor,
        data_dir,
        split,
        tasks,
        rank=0,
        world_size=1,
        max_samples=None,
        min_duration=0.1,
        max_duration=60.0,
        use_aux_clf=False,
        specaugment_mode="clean_only",
        zero_shot_inference=False,
    ):
        self.tasks = tasks
        self.data = None
        self.use_aux_clf = use_aux_clf
        self.specaugment_mode = specaugment_mode
        self.zero_shot_inference = zero_shot_inference

        data_list = []
        files = list((Path(data_dir) / split).rglob("*.parquet"))
        if rank == 0:
            logger.info(
                f"[SAPTestDataset] Loading data for tasks {tasks}. Found {len(files)} parquet files."
            )

        def filter_fn(example, _task="asr"):
            value = example[_task]
            if value is None:
                return False
            if isinstance(value, (int, float)):
                return True
            if isinstance(value, str) and len(value.strip()) > 0:
                return True
            return False

        for task in tasks:
            _data = load_dataset(
                data_dir, data_files={f"{split}": [str(f) for f in files]}
            )[split]
            if task != "asr":
                clf_col = TASK2CLFCOL_MAP[task]
                if isinstance(clf_col, tuple):
                    d = [
                        _data.filter(
                            functools.partial(filter_fn, _task=c),
                            desc="Empty value filter",
                        )
                        for c in clf_col
                    ]
                    _data = concatenate_datasets(d)
                else:
                    _data = _data.filter(
                        functools.partial(filter_fn, _task=clf_col),
                        desc="Empty value filter",
                    )
            if rank == 0:
                logger.info(
                    f"[SAPTestDataset] Rows for task '{task}' (after filtering empty examples): {_data.num_rows}"
                )
            _data = _data.add_column("task", [task] * len(_data))
            data_list.append(_data)
        self.data = concatenate_datasets(data_list)
        if rank == 0:
            logger.info(
                f"[SAPTestDataset] Concatenated test dataset: {self.data.num_rows}"
            )

        if max_samples is not None:
            self.data = self.data.select(range(max_samples))

        self.data = self.data.remove_columns(
            [
                col
                for col in self.data.column_names
                if col not in COLUMNS_TO_KEEP + ["task"]
            ]
        )
        self.data = self.data.cast_column(
            "wav", datasets.features.Audio(sampling_rate=16_000)
        )
        self.data = self.data.filter(
            lambda x: min_duration
            <= len(x["wav"]["array"]) / x["wav"]["sampling_rate"]
            <= max_duration,
            desc="Audio duration filter",
        )
        if rank == 0:
            logger.info("[SAPTestDataset] Final test dataset after duration filtering:")
            logger.info(self.data)
        self.processor = processor

        if world_size > 1:
            self.data = self.data.shard(world_size, rank)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        task = data["task"]
        if self.zero_shot_inference:
            instruction = ZERO_SHOT_INSTRUCTION[task]
        else:
            instruction = INSTRUCTION[task]
        clf_str = get_clf_string(data, task)

        user_message = {
            "role": "user",
            "content": "<|audio_1|>\n" + instruction,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )

        additional_kwargs = {}
        if isinstance(
            self.processor.audio_processor, Phi4MMSpecAugmentAudioFeatureExtractor
        ):
            additional_kwargs["specaugment_mode"] = self.specaugment_mode
        inputs = self.processor(
            text=prompt,
            audios=[(data["wav"]["array"], data["wav"]["sampling_rate"])],
            return_tensors="pt",
            **additional_kwargs,
        )
        transcript = data["words"] if "asr" in task else ""
        answer = f"{transcript}{clf_str}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors="pt").input_ids

        input_ids = inputs.input_ids
        labels = answer_ids

        # For aux clf extract cls_labels
        aux_clf_dict = {}
        if self.use_aux_clf:
            if task != "asr":
                try:
                    candidate = clf_str.replace(CLF_SPECIAL_TOKEN, "").strip()
                    combined_mappings = ETIOLOGY_MAP | CATEGORY_MAP
                    for k, v in combined_mappings.items():
                        if candidate == k:
                            candidate = v + 1
                            break
                    clf_score = int(candidate) - 1
                    aux_clf_dict["cls_labels"] = clf_score
                except Exception as e:
                    logger.error(
                        f"Unable to obtain integer value from classification string '{clf_str}' ({task=}). "
                        f"Falling back to default value. {e}"
                    )
            else:
                aux_clf_dict["cls_labels"] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
            "tasks": task,
            **aux_clf_dict,
        }


class SAPTSVInferenceDataset(Dataset):
    def __init__(
        self,
        processor,
        inference_args,
        rank=0,
        world_size=1,
        min_duration=1.0,
        max_duration=30.0,
        vad_model=None,
        zero_shot_inference=False,
    ):
        self.data = load_sapc_tsv_test_set(inference_args, vad_model=vad_model)
        if inference_args.max_test_samples is not None:
            self.data = self.data.shuffle(seed=424242)
            self.data = self.data.select(range(inference_args.max_test_samples))

        self.data = self.data.remove_columns(
            [col for col in self.data.column_names if col not in COLUMNS_TO_KEEP]
        )
        self.data = self.data.cast_column(
            "wav", datasets.features.Audio(sampling_rate=16_000)
        )
        self.data = self.data.filter(
            lambda x: min_duration
            <= len(x["wav"]["array"]) / x["wav"]["sampling_rate"]
            <= max_duration
        )
        logger.info("Dataset after filtering:")
        logger.info(self.data)
        self.processor = processor
        # self.task = inference_args.task
        self.task = "asr"
        if zero_shot_inference:
            self.instruction = ZERO_SHOT_INSTRUCTION[self.task]
        else:
            self.instruction = INSTRUCTION[self.task]

        if world_size > 1:
            self.data = self.data.shard(world_size, rank)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Required fields: "wav", "utt_id", "intelligbility", "intelligibility", "naturalness", "words", "etiology", "category"
        This will use answer = f"{transcript}{clf_str}{ANSWER_SUFFIX}" as labels.
        This means we will have <|clf|> followed by the classifier label in the label ids.
        """
        data = self.data[idx]
        clf_str = get_clf_string(data, self.task)
        # No rating available for this example, so we fall back to just doing ASR
        needs_fallback_instruction = len(clf_str) < 1
        if needs_fallback_instruction:
            self.instruction = INSTRUCTION["asr"]

        user_message = {
            "role": "user",
            "content": "<|audio_1|>\n" + self.instruction,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            audios=[(data["wav"]["array"], data["wav"]["sampling_rate"])],
            return_tensors="pt",
        )

        return {
            "input_ids": inputs.input_ids,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
            "tasks": self.task,
        }


class PDGermanInferenceDataset(Dataset):
    def __init__(
        self,
        processor,
        inference_args,
        split="train",
        rank=0,
        world_size=1,
        min_duration=1.0,
        max_duration=30.0,
        vad_model=None,
        zero_shot_inference=False,
    ):
        data_dir = inference_args.data_dir
        self.data = load_dataset(data_dir, split=split)
        if inference_args.max_test_samples is not None:
            self.data = self.data.shuffle(seed=424242)
            self.data = self.data.select(range(inference_args.max_test_samples))

        self.data = self.data.remove_columns(
            [col for col in self.data.column_names if col not in COLUMNS_TO_KEEP]
        )
        self.data = self.data.cast_column(
            "wav", datasets.features.Audio(sampling_rate=16_000)
        )
        self.data = self.data.filter(
            lambda x: min_duration
            <= len(x["wav"]["array"]) / x["wav"]["sampling_rate"]
            <= max_duration
        )
        logger.info("PD_DE dataset after filtering:")
        logger.info(self.data)
        self.processor = processor
        self.task = "asr"
        if zero_shot_inference:
            self.instruction = ZERO_SHOT_INSTRUCTION[self.task]
        else:
            self.instruction = INSTRUCTION[self.task]

        if world_size > 1:
            self.data = self.data.shard(world_size, rank)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        clf_str = get_clf_string(data, self.task)
        # No rating available for this example, so we fall back to just doing ASR
        needs_fallback_instruction = len(clf_str) < 1
        if needs_fallback_instruction:
            self.instruction = INSTRUCTION["asr"]

        transcript = data["words"] if "asr" in self.task else ""
        answer = f"{transcript}{clf_str}{ANSWER_SUFFIX}"
        labels = self.processor.tokenizer(answer, return_tensors="pt").input_ids
        utt_id = data["utt_id"]

        user_message = {
            "role": "user",
            "content": "<|audio_1|>\n" + self.instruction,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            audios=[(data["wav"]["array"], data["wav"]["sampling_rate"])],
            return_tensors="pt",
        )

        return {
            "input_ids": inputs.input_ids,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
            "labels": labels,
            "tasks": self.task,
            "utt_ids": utt_id,
        }
