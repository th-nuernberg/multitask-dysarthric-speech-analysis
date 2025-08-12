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
import torch
import logging
from transformers import BatchFeature

logger = logging.getLogger(__name__)


def pad_sequence(sequences, padding_side="right", padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(t.dim() == ndim for t in tensors[1:]), (
        "All tensors must have the same number of dimensions"
    )

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


def collate_fn(batch):
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    tasks = []
    for inputs in batch:
        input_ids_list.append(inputs["input_ids"][0])
        labels_list.append(inputs["labels"][0])
        input_audio_embeds_list.append(inputs["input_audio_embeds"])
        tsk = inputs.get("tasks", None)
        if tsk is not None:
            tasks.append(tsk)
        audio_embed_sizes_list.append(inputs["audio_embed_sizes"])
        audio_attention_mask_list.append(
            inputs["input_audio_embeds"].new_full(
                (inputs["input_audio_embeds"].size(1),), True, dtype=torch.bool
            )
        )
    try:
        input_ids = pad_sequence(input_ids_list, padding_side="left", padding_value=0)
        labels = pad_sequence(labels_list, padding_side="left", padding_value=0)
        audio_attention_mask = (
            pad_sequence(
                audio_attention_mask_list, padding_side="right", padding_value=False
            )
            if len(audio_attention_mask_list) > 1
            else None
        )
    except Exception as e:
        logger.error(e)
        logger.error(input_ids_list)
        logger.error(labels_list)
        raise
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    add_dict = {}
    if len(tasks) > 0:
        add_dict["tasks"] = tasks

    return BatchFeature(
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_attention_mask": audio_attention_mask,
            "input_mode": 2,  # speech mode
            **add_dict,
        }
    )


def custom_collate_fn(batch):
    input_ids_list = []
    labels_list = []
    clf_label_list = []
    input_audio_embeds_list = []
    input_audio_embeds_other_list = []

    aux_reg_labels_list = []
    aux_reg_mask_list = []
    aux_reg_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    tasks = []
    for inputs in batch:
        input_ids_list.append(inputs["input_ids"][0])
        labels_list.append(inputs["labels"][0])
        input_audio_embeds_list.append(inputs["input_audio_embeds"])

        other_embeds = inputs.get("input_audio_embeds_other", None)
        if other_embeds is not None:
            input_audio_embeds_other_list.append(other_embeds)

        aux_reg_labels = inputs.get("aux_reg_labels", None)
        if aux_reg_labels is not None:
            aux_reg_labels_list.append(aux_reg_labels)

        aux_reg_mask = inputs.get("aux_reg_mask", None)
        if aux_reg_mask is not None:
            aux_reg_mask_list.append(aux_reg_mask)

        aux_reg_embeds = inputs.get("aux_reg_embeds", None)
        if aux_reg_embeds is not None:
            aux_reg_embeds_list.append(aux_reg_embeds)

        tsk = inputs.get("tasks", None)
        if tsk is not None:
            tasks.append(tsk)

        lbl = inputs.get("cls_labels", None)
        if lbl is not None and tsk is not None:
            clf_label_list.append(lbl)
        audio_embed_sizes_list.append(inputs["audio_embed_sizes"])
        audio_attention_mask_list.append(
            inputs["input_audio_embeds"].new_full(
                (inputs["input_audio_embeds"].size(1),), True, dtype=torch.bool
            )
        )

    try:
        input_ids = pad_sequence(input_ids_list, padding_side="left", padding_value=0)
        labels = pad_sequence(labels_list, padding_side="left", padding_value=0)
        audio_attention_mask = (
            pad_sequence(
                audio_attention_mask_list, padding_side="right", padding_value=False
            )
            if len(audio_attention_mask_list) > 1
            else None
        )
    except Exception as e:
        logger.error(e)
        logger.error(input_ids_list)
        logger.error(labels_list)
        raise
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    add_dict = {}
    if len(tasks) > 0:
        add_dict["tasks"] = tasks
    if len(input_audio_embeds_other_list) > 0:
        input_audio_embeds_other = cat_with_pad(input_audio_embeds_other_list, dim=0)
        add_dict["input_audio_embeds_other"] = input_audio_embeds_other
    if len(clf_label_list) > 0:
        add_dict["cls_labels"] = torch.LongTensor(clf_label_list)
    if len(aux_reg_labels_list) > 0:
        add_dict["aux_reg_labels"] = torch.stack(aux_reg_labels_list)
    if len(aux_reg_mask_list) > 0:
        add_dict["aux_reg_mask"] = torch.stack(aux_reg_mask_list)
    if len(aux_reg_embeds_list) > 0:
        add_dict["aux_reg_embeds"] = torch.stack(aux_reg_embeds_list)

    return BatchFeature(
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_attention_mask": audio_attention_mask,
            "input_mode": 2,  # speech mode
            **add_dict,
        }
    )


def sapc_inference_collate_fn(batch):
    """This is the same as collate_fn with the only difference being that it doesn't produce labels"""
    input_ids_list = []
    input_audio_embeds_list = []
    input_audio_embeds_other_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    tasks = []
    for inputs in batch:
        input_ids_list.append(inputs["input_ids"][0])
        if inputs.get("input_audio_embeds_other", None) is not None:
            input_audio_embeds_other_list.append(inputs["input_audio_embeds_other"])
        input_audio_embeds_list.append(inputs["input_audio_embeds"])
        audio_embed_sizes_list.append(inputs["audio_embed_sizes"])
        audio_attention_mask_list.append(
            inputs["input_audio_embeds"].new_full(
                (inputs["input_audio_embeds"].size(1),), True, dtype=torch.bool
            )
        )
        tsk = inputs.get("tasks", None)
        if tsk is not None:
            tasks.append(tsk)

    try:
        input_ids = pad_sequence(input_ids_list, padding_side="left", padding_value=0)
        audio_attention_mask = (
            pad_sequence(
                audio_attention_mask_list, padding_side="right", padding_value=False
            )
            if len(audio_attention_mask_list) > 1
            else None
        )
    except Exception as e:
        logger.error(e)
        logger.error(input_ids_list)
        raise
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    add_dict = {}
    if len(tasks) > 0:
        add_dict["tasks"] = tasks
    if len(input_audio_embeds_other_list) > 0:
        input_audio_embeds_other = cat_with_pad(input_audio_embeds_other_list, dim=0)
        add_dict["input_audio_embeds_other"] = input_audio_embeds_other
    return BatchFeature(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_attention_mask": audio_attention_mask,
            "input_mode": 2,  # speech mode
            **add_dict,
        }
    )


def pd_de_inference_collate_fn(batch):
    """Same as collate_fn but it also returns utt_ids"""
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    tasks = []
    utt_ids = []
    for inputs in batch:
        input_ids_list.append(inputs["input_ids"][0])
        labels_list.append(inputs["labels"][0])
        input_audio_embeds_list.append(inputs["input_audio_embeds"])
        tsk = inputs.get("tasks", None)
        if tsk is not None:
            tasks.append(tsk)

        utt = inputs.get("utt_ids", None)
        if utt is not None:
            utt_ids.append(utt)
        audio_embed_sizes_list.append(inputs["audio_embed_sizes"])
        audio_attention_mask_list.append(
            inputs["input_audio_embeds"].new_full(
                (inputs["input_audio_embeds"].size(1),), True, dtype=torch.bool
            )
        )
    try:
        input_ids = pad_sequence(input_ids_list, padding_side="left", padding_value=0)
        labels = pad_sequence(labels_list, padding_side="left", padding_value=0)
        audio_attention_mask = (
            pad_sequence(
                audio_attention_mask_list, padding_side="right", padding_value=False
            )
            if len(audio_attention_mask_list) > 1
            else None
        )
    except Exception as e:
        logger.error(e)
        logger.error(input_ids_list)
        logger.error(labels_list)
        raise
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    add_dict = {}
    if len(tasks) > 0:
        add_dict["tasks"] = tasks
    if len(utt_ids) > 0:
        add_dict["utt_ids"] = utt_ids

    return BatchFeature(
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes": audio_embed_sizes,
            "audio_attention_mask": audio_attention_mask,
            "input_mode": 2,  # speech mode
            **add_dict,
        }
    )
