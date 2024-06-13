# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import sys
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
import cv2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import random
local_rank = None
custom_args = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                # Wei: allow multiple image token placeholders in a single sentence
                if sentence['value'].count(DEFAULT_IMAGE_TOKEN) > 1:
                    pass
                else:  #  move <image>\n into the beginning of the sentence
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

from llava.mm_utils import safe_dec
def alternative_masking(tokenizer, labels, sep, mask_start='USER:', mask_end='ASSISTANT:'):
    cur_str = ''
    mask_mode = True
    sep_ix = -1
    for ix, tok in enumerate(labels):
        cur_str += safe_dec(tokenizer, tok)
        if cur_str.endswith(mask_start):
            mask_mode = True
            if sep_ix > 0:
                for j in range(sep_ix, ix):
                    labels[j] = IGNORE_INDEX # mask the mask_start
        if cur_str.endswith(mask_end):
            labels[ix] = IGNORE_INDEX
            mask_mode = False
        if cur_str.endswith(sep):
            sep_ix = ix + 1
        if mask_mode:
            labels[ix] = IGNORE_INDEX
    return labels

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            # leo was here
            conv.append_message(role, sentence["value"], negs=(sentence["negs"] if "negs" in sentence else None))
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        tok_conv = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
        _tok_conv_inp_ids = [x[0] for x in tok_conv]
        input_ids = torch.stack(_tok_conv_inp_ids, dim=0)
        negs = [x[1] for x in tok_conv]
    else:
        # leo: here we ignore negatives if they exist
        _conversations = [x[0] for x in conversations]
        input_ids = tokenizer(
            _conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        negs = None

    targets = input_ids.clone()
    # print(f'### {targets.shape}')

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    _debug_flag = False

    # if 'labrador' in tokenizer.name_or_path.lower():
    #     assistant_len = 1 if 'GPTNeoXTokenizerFast' in str(type(tokenizer)) else 1  # leo: was 2 in orig llava
    #     extra_len = 0 if 'GPTNeoXTokenizerFast' in str(type(tokenizer)) else 0  # leo: was 0 in orig llava
    #     start_len = 0 if 'GPTNeoXTokenizerFast' in str(type(tokenizer)) else 0  # leo: was 1 in orig llava
    #     end_append = conv.sep2 if 'GPTNeoXTokenizerFast' in str(type(tokenizer)) else conv.sep2 # '<|endoftext|>'  # leo: was '' in orig llava
    # else:
    #     assistant_len = 1 if 'GPTNeoXTokenizerFast' in str(type(tokenizer)) else 2  # leo: was 2 in orig llava
    #     extra_len = 0 if 'GPTNeoXTokenizerFast' in str(type(tokenizer)) else 0  # leo: was 0 in orig llava
    #     start_len = 0 if 'GPTNeoXTokenizerFast' in str(type(tokenizer)) else 1  # leo: was 1 in orig llava
    #     end_append = conv.sep2 if 'GPTNeoXTokenizerFast' in str(type(tokenizer)) else ''  # leo: was '' in orig llava

    assistant_len = 1 if 'GPTNeoXTokenizerFast' in str(type(tokenizer)) else 2  # leo: was 2 in orig llava
    extra_len = 0 if 'GPTNeoXTokenizerFast' in str(type(tokenizer)) else 0  # leo: was 0 in orig llava
    start_len = 0 if 'GPTNeoXTokenizerFast' in str(type(tokenizer)) else 1  # leo: was 1 in orig llava
    end_append = conv.sep2 if 'GPTNeoXTokenizerFast' in str(type(tokenizer)) else ''  # leo: was '' in orig llava

    # debug: ''.join([safe_dec(tokenizer, x) for x in alternative_masking(tokenizer, tokenizer_image_token(conversation, tokenizer), conv.sep2)])

    # Mask targets
    #leo: this should be left unchanged
    conversations = [x[0] for x in conversations]
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        if 'labrador' in tokenizer.name_or_path.lower():
            alternative_masking(tokenizer, target, conv.sep2)
        else:
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = start_len
            target[:cur_len] = IGNORE_INDEX #  set the first token (start token ) to -100
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)  #  the separator is "ASSISTANT: "  ,  the parts[0] is the instruction
                if len(parts) != 2:
                    break
                parts[0] += sep

                expected_round_text = rou + (end_append if len(rou) > 0 else '')

                if has_image:
                    round_len = len(tokenizer_image_token(expected_round_text, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - assistant_len  #  the 2 tokens are for  sep ("ASSISTANT:")
                else:
                    round_len = len(tokenizer(expected_round_text).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - assistant_len

                if _debug_flag:
                    from llava.mm_utils import safe_dec
                    targ_str = ''
                    targ_src = target[cur_len : cur_len + round_len + (extra_len if round_len!=0 else 0)]
                    for x in targ_src:
                        targ_str += safe_dec(tokenizer, x)
                    exp_str = ''
                    if has_image:
                        exp_src = tokenizer_image_token(expected_round_text, tokenizer)
                    else:
                        exp_src = tokenizer(expected_round_text).input_ids
                    for x in exp_src:
                        exp_str += safe_dec(tokenizer, x)
                    print(f'Expected: {exp_str}')
                    print(f'Actual  : {targ_str}')
                    print(f'TE: {exp_src}')
                    print(f'TA: {targ_src.cpu().numpy().tolist()}')

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len + (extra_len if round_len!=0 else 0)
            target[cur_len:] = IGNORE_INDEX   #  concatenate tokens of all instruction-response pair rounds , and set the instruction tokens to -100

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    return dict(
        input_ids=input_ids,
        labels=targets,   #  concatenate tokens of all instruction-response pair rounds , and set the instruction tokens to -100,  image place is -200
        negs=negs
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def fuse_scot(img_path, _sources):
    sources = _sources
    if custom_args.scot_type is not None:
        assert len(sources) == 1
        assert len(img_path) == 1
        img_path = img_path[0]  # it is always a list, we currently support only the basic llava form with a single image, also gif fix should not be on
        scot_modifiers = custom_args.scot_type.split('_')
        if 'mask' == scot_modifiers[0]: #mask_16x16, mask_tight_16x16
            mask_sz = [int(x) for x in scot_modifiers[-1].split('x')]
            anno_tight = ((len(scot_modifiers) > 2) and (scot_modifiers[1] == 'tight'))
            anno_box = ((len(scot_modifiers) > 2) and (scot_modifiers[1] == 'box'))
            meta_path = os.path.join(custom_args.metadata_path, img_path)
            if os.path.exists(meta_path) and os.path.exists(os.path.join(meta_path, 'label.json')) and os.path.exists(os.path.join(meta_path, '_mask.npy')):
                with open(os.path.join(meta_path, 'label.json'), 'r') as f:
                    labels = json.load(f)
                # mask = Image.open(os.path.join(meta_path, 'mask.jpg')).convert("RGB")
                orig_img = cv2.imread(os.path.join(meta_path, 'raw_image.jpg'))
                orig_sz_yx = orig_img.shape[:2]
                #bbox = [x, y, x y] - TL - BR, absolute in raw image size
                with open(os.path.join(meta_path, '_mask.npy'), 'rb') as f:
                    mask = np.load(f)
                mask_ = cv2.resize(mask, mask_sz, 0, 0, interpolation = cv2.INTER_NEAREST)
                if not np.all(np.array(orig_sz_yx) == np.array(mask.shape)):
                    print(f'WARNING: size mismatch, orig_sz_yx={orig_sz_yx}, mask.shape={mask.shape}, meta_path: {meta_path}')
                scaleXY = np.array(orig_sz_yx) / np.array(mask_.shape)
                scaleXY = scaleXY[[1, 0, 1, 0]]
                h_turn = {'from': 'human', 'value': '<image>\nDensely annotate the image.'}
                g_turn = {'from': 'gpt'}
                j_labels = []
                for lbl in labels['mask']:
                    if 'box' in lbl:
                        lbl['box'] = [f'{(x/y):.2f}' for x, y in zip(lbl['box'], scaleXY)]
                    j_labels.append(lbl)
                if anno_tight or anno_box:
                    anno = {}
                    boxes = {}
                    for lbl in j_labels:
                        anno[lbl['value']] = lbl['label']
                        if anno_box and ('box' in lbl):
                            boxes[lbl['value']] = json.dumps(lbl['box']).replace('"','')
                    j_labels = anno
                s_mask = '[\n'
                for r in mask_:
                    s_mask += '[' + ', '.join([str(int(x)) for x in r]) + ']\n'
                s_mask += ']'
                if not anno_box:
                    g_value = {'entities': j_labels, 'mask': s_mask}
                else:
                    g_value = {'entities': j_labels, 'boxes': boxes}
                g_turn['value'] = json.dumps(g_value)
                # if not anno_tight:
                #     g_turn['value'] = json.dumps(g_value, indent=1)
                # else:
                #     g_turn['value'] = json.dumps(g_value)

                # update the conversation with an extra turn
                sources[0][0]['value'] = sources[0][0]['value'].replace('<image>\n', '').replace('<image>', '')
                sources[0] = [h_turn] + [g_turn] + sources[0]
    return sources


def process_mimic_sd_dataset(dataset_dir):
    prefix = 'Image 1: <image>\nImage 2: <image>\n'
    dataset = 'SD'
    ds = load_dataset(dataset_dir)
    train_dataset = ds['train']
    dataset_llava_format = []
    list_of_set_of_instructions = []
    ids_that_are_grouped = []
    all_sample_dict = {}
    for sample_idx, sample in enumerate(tqdm(train_dataset)):
        id = sample['id']
        all_sample_dict[id] = sample
        related_instructions = sample['related instructions']
        set_of_instructions = []
        if id not in ids_that_are_grouped:
            set_of_instructions.append(id)
            ids_that_are_grouped.append(id)
            for related_instruction in related_instructions:
                set_of_instructions.append(related_instruction)
                ids_that_are_grouped.append(related_instruction)
            list_of_set_of_instructions.append(set_of_instructions)
        else:
            continue
    for merged_sample_id, set_of_instructions in enumerate(tqdm(list_of_set_of_instructions)):
        conversations = []
        len_conversations = len(set_of_instructions)
        for conv_id in range(len_conversations):
            if conv_id == 0:
                prefix_ = prefix
            else:
                prefix_ = ''
            conversations.append({
                'from': 'human',
                'value': prefix_ + all_sample_dict[set_of_instructions[conv_id]]['instruction']
            })
            conversations.append({
                'from': 'gpt',
                'value': all_sample_dict[set_of_instructions[conv_id]]['answer']
            })
        dataset_llava_format.append({
            'id': f'{dataset}_{merged_sample_id}',
            'image': all_sample_dict[set_of_instructions[0]]['images'],
            'conversations': conversations
        })
    return dataset_llava_format

def process_idefics_cgd_dataset(dataset_dir):
    prefix = 'Image 1: <image>\nImage 2: <image>\n'
    dataset = 'mimic_cgd'
    ds = load_dataset(dataset_dir)
    train_dataset = ds['train']
    dataset_llava_format = []
    for sample_id, sample in enumerate(tqdm(train_dataset)):
        conversations = []
        for conv_id in range(len(sample['texts'])):
            if conv_id == 0:
                prefix_ = prefix
            else:
                prefix_ = ''
            conversations.append({
                'from': 'human',
                'value': prefix_ + sample['texts'][conv_id]['user']
            })
            conversations.append({
                'from': 'gpt',
                'value': sample['texts'][conv_id]['assistant']
            })

        dataset_llava_format.append({
            'id': f'{dataset}_{sample_id}',
            'image': sample['images'],  # a list of PIL images
            'conversations': conversations
        })
    return dataset_llava_format
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        if ',' in data_path:
            data_path_list = data_path.split(',')
            list_data_dict = []
            for data_path in data_path_list:  # load each json file
                if data_path.endswith('.json'):
                    loaded_data = json.load(open(data_path, "r"))# list of sample dict,  each dict has  id, image, conversations
                elif 'mimic_sd_dataset/SD' in data_path:  # HF dataset
                    loaded_data = process_mimic_sd_dataset(data_path)
                elif 'idefics_dataset/mimic_cgd' in data_path:
                    loaded_data = process_idefics_cgd_dataset(data_path)
                else:
                    raise ValueError(f'Unknown dataset: {data_path}')
                list_data_dict += loaded_data
            # random.shuffle(list_data_dict)
        else:
            if data_path.endswith('.json'):
                loaded_data = json.load(
                    open(data_path, "r"))  # list of sample dict,  each dict has  id, image, conversations
            elif 'mimic_sd_dataset/SD' in data_path:  # HF dataset
                loaded_data = process_mimic_sd_dataset(data_path)
            elif 'idefics_dataset/mimic_cgd' in data_path:
                loaded_data = process_idefics_cgd_dataset(data_path)
            else:
                raise ValueError(f'Unknown dataset: {data_path}')
            list_data_dict = loaded_data

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            _image_file = self.list_data_dict[i]['image']  #  a list of strings
            _image_folder = self.data_args.image_folder.split(',')
            processor = self.data_args.image_processor

            #Leo: introduce multi-image per conversation support
            _image_file = (_image_file if isinstance(_image_file, list) else [_image_file]) # can be either list or string
            _image = []
            for image_file in _image_file: #  here image_file is a string
                # leo-fix
                if isinstance(image_file, str):
                    for image_folder in _image_folder:
                        img_path = os.path.join(image_folder, image_file)
                        if not os.path.exists(img_path):
                            # attempt .jpg --> .gif or any similar replacement based on what is there
                            orig_image_file = image_file
                            parts = image_file.split('.')
                            parts[1]='*'
                            import glob
                            options = glob.glob(os.path.join(image_folder, '.'.join(parts)))
                            #image_file = os.path.basename(options[0])
                            #print(f'{orig_image_file}-->{image_file}, image_folder={image_folder}')
                            if len(options) > 0:
                                img_path = options[0]
                                break
                        else:
                            break

                    image = Image.open(img_path).convert('RGB')  # PIL image
                elif isinstance(image_file, Image.Image):
                    image = image_file
                else:
                    raise ValueError(f'Unknown image type: {type(image_file)}')
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean)) # (640, 427) -> (640, 640)
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0] # (640, 640) -> (336, 336)
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                _image.append(image)

            # Leo: introduce multi-image per conversation support
            image = _image #now image is a list of all conversation images, can be more than one

            sources = preprocess_multimodal(
                fuse_scot(_image_file, copy.deepcopy([e["conversations"] for e in sources])),
                self.data_args) #  a list of 6 (3 QA pairs )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i])) #  concatenation of [ system_prompt, -200, instruction1 , response1, -200, instruction2, response2, -200, instruction3, response3]
        if isinstance(i, int):
            # print(data_dict.keys())
            if ('negs' in data_dict) and (data_dict["negs"] is not None):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0],
                                 negs=data_dict["negs"][0])
                # print((len(data_dict['negs']), data_dict['input_ids'].shape))
            else:
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        if 'negs' in instances[0]:
        #     print('### negs in collator')
        #     print(len(instances))
        #     for inst in instances:
        #         print((len(inst['negs']), inst['input_ids'].shape))
            negs = [(x['negs'] if 'negs' in x else None) for x in instances]
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                negs=negs
            )
        else:
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )

        if 'image' in instances[0]:
            _images = [instance['image'] for instance in instances]

            # Leo: added multi-image support for conversations, to support ICL and other multi-image use-cases
            images = []
            for x in _images:
                if isinstance(x, list):
                    images.extend(x)
                else:
                    images.append(x)

            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


@dataclass
class CustomArguments:
    scot_type: Optional[str] = field(
        default=None,
        metadata={"help": "defines both type and size hyperparameter in <type>_<H>x<W> format"},
    )
    metadata_path: Optional[str] = field(
        default='./metadata',
        metadata={"help": "matches the ./data folder structure, each image relative path is a folder with metadata"},
    )
    loss_type: Optional[str] = field(
        default='clm',
        metadata={"help": "choose one of: clm, neg, focal, or neg_focal."},
    )
    num_top_focal: Optional[int] = field(
        default=3,
        metadata={"help": "top-k for focal loss and printing."},
    )
    num_rand_neg: Optional[int] = field(
        default=10,
        metadata={"help": "num random points for focal loss in rand_focal."},
    )
    print_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "for printing batch samples."},
    )
    do_debug: Optional[bool] = field(
        default=False,
        metadata={"help": "for debugging."},
    )
    debug_port: Optional[int] = field(
        default=12345,
        metadata={"help": "for debugging."},
    )
    loss_coeff: Optional[float] = field(
        default=1.0,
        metadata={"help": "coefficient for the additional loss"},
    )

def train():
    global local_rank
    global custom_args

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, CustomArguments))
# <<<<<<< HEAD
#     model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()  # xx
# =======
    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    if custom_args.do_debug:
        from cvar_pyutils.debugging_tools import set_remote_debugger
        set_remote_debugger('9.106.159.216', custom_args.debug_port) # '9.67.29.115'

# >>>>>>> leonid
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif 'granite' in model_args.model_name_or_path.lower():
            model = LlavaGraniteForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
            from ibm_models import AttentionImplementation
            model.inject_attention_implementation(AttentionImplementation.flash)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,    #  load the Vicuna model
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )

    if '_for_finetune' in model_args.model_name_or_path:
        mm_projector_weights = torch.load(os.path.join(model_args.model_name_or_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        ) # LoraConfig  lora_alpha 256, lora_dropout 0.05, r 128, target_modules,
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)  # here model is PeftModelForCausalLM

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    elif 'granite' in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,  #  the path is of Vicuna model
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:   #  load the vision tower,  here also pretrain_mm_mlp_adapter is loaded in  initialize_vision_modules
        model.get_model().initialize_vision_modules(   # modified by leonid
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter  # if tune_mm_mlp_adapter is True, this is the first stage, only the mm_projector is trained
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,   #  model.base_model.model (LLavaLlamaForCausalLM) has 3 components: base_model, vision_tower, and mm_projector
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
