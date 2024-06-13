#    Copyright 2023 Haotian Liu
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
import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from collections import OrderedDict
import os
class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args) # modified by leonid,  contains vision_tower.latt_proj   linear layer from 5120 to 1024

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # print(mm_projector_weights)
            # warnings.warn(f'{mm_projector_weights}')
            try:
                loaded_weight = get_w(mm_projector_weights, 'mm_projector')
                self.mm_projector.load_state_dict(loaded_weight)

                # new_loaded_weight = OrderedDict()
                # key_list = ['0.weight', '0.bias', '2.weight', '2.bias']
                # for key in key_list:
                #     new_loaded_weight[key] = loaded_weight[key]
                # print(new_loaded_weight)
                # warnings.warn(f'{new_loaded_weight}')
                # for key in new_loaded_weight:
                #     warnings.warn(f'{key}:{new_loaded_weight[key].shape}')
                # warnings.warn(f'{self.mm_projector}')
                # check if the shape matches
                # for key in new_loaded_weight:
                #     if self.mm_projector.state_dict()[key].shape != new_loaded_weight[key].shape:
                #         warnings.warn(f'{key} shape mismatch: {self.mm_projector.state_dict()[key].shape} vs {new_loaded_weight[key].shape}')
                # self.mm_projector.load_state_dict( new_loaded_weight)
                # warnings.warn(f'{self.mm_projector.parameters()}')
            except:
                warnings.warn('mm_projector failed to load! Likely a size mismatch due trying transfering mm_projector from a different model?')

def simple_count(x):
    cnt = 0
    for y in x:
        for z in y:
            cnt += len(z)
    return cnt

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, texts=None):
        image_features = self.get_model().get_vision_tower()(images, texts, llm=self.get_model(), tokenizer=self.llm_tokenizer)
        image_features = self.get_model().mm_projector(image_features)
        print(torch.abs(self.get_model().mm_projector[0].weight).sum().item())
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, negs=None
    ):

        do_negs = (negs is not None)
        if do_negs:
            # protect from mal-formed negs
            for iN in range(len(negs)):
                if negs[iN] is None:
                    negs[iN] = [[] for _ in range(labels.shape[1])]
                for iL in range(len(negs[iN])):
                    if negs[iN][iL] is None:
                        negs[iN][iL] = []

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, negs

        if self.get_model().get_vision_tower().latt_model_name is not None:
            T = [[y for y in x.tolist() if y != -200] for x in input_ids]
            T = [self.llm_tokenizer.decode(x) for x in T]
            T = [x.split('USER:') for x in T]
            T = [[x.split('ASSISTANT:')[0].strip() for x in y[1:]] for y in T]
            texts = T
        else:
            texts = None

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images, texts)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
# <<<<<<< HEAD
#             image_features = self.encode_images(images).to(self.device) # (3,3, 336, 336) -> (3, 576, 4096)
# =======
            image_features = self.encode_images(images, texts).to(self.device)
# >>>>>>> leonid

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        if do_negs:  #  only keep the negative tokens in the response tokens
            # print(f'$$$$$$$$ negs prior count = {simple_count(negs)} $$$$$$$$')
            negs = [[x for x, y in zip(cur_negs, cur_attention_mask) if y] for cur_negs, cur_attention_mask in zip(negs, attention_mask)]

        new_input_embeds = []
        new_labels = []
        if do_negs:
            new_negs = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):  # cur_input_ids: system_prompt token, img placeholder = -200, instruction1 token, response1 token, instruction2 token, response2 token ...
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                if do_negs:
                    new_negs.append(negs[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx] # #  only the response tokens are not -100, the rest (system_promt, image placeholder, instructions) are -100
            if do_negs:
                cur_negs = negs[batch_idx]
            cur_labels_noim = []
            if do_negs:
                cur_negs_noim = []
            for i in range(len(image_token_indices) - 1): # split the input_ids and labels into several segments with the separator being the image placeholder
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                if do_negs:
                    cur_negs_noim.append(cur_negs[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim] # [35, 519]   total length = 35 + 1 (img placeholder) +519 = 555
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)) # (554, 4096)
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) # split into [(35, 4096), (519, 4096)]
            cur_new_input_embeds = []
            cur_new_labels = []
            if do_negs:
                cur_new_negs = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if do_negs:
                    cur_new_negs.extend(cur_negs_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]  # (576, 4096)  each image has 576 tokens, each token is 4096 dim
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    if do_negs:
                        cur_new_negs.extend([[] for _ in range(cur_image_features.shape[0])]) # add placeholder neg token for image tokens
            # cur_new_input_embeds [(35, 4096) system prompt, (576, 4096) image, (519, 4096) instruction1 token, response1 token, instruction2 token, response2 token ... ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds) # (1130, 4096)
            cur_new_labels = torch.cat(cur_new_labels)#  [(35, )  system prompt all -100, (576, ) image all -100, (519, ) instruction1 token all -100, response1 token, instruction2 token all -100, response2 token ...  ]
            if do_negs:
                cur_new_negs = cur_new_negs # this should be ok,   for each token in the sequence, we have a list of neg tokens (for most tokens, the list is empty)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            if do_negs:
                new_negs.append(cur_new_negs)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None: # tokenizer_model_max_length  2048
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            if do_negs:
                new_negs = [x[:tokenizer_model_max_length] for x in new_negs]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        # padding for all samples in a batch to max_len,  for labels pad with -100, for others pad with 0
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device) # (bz, max_len)  padding for all samples in a batch to max_len
        if do_negs:
            new_negs_padded = [[[] for ii in range(max_len)] for jj in range(batch_size)]
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)  # (bz, max_len)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device) # (bz, max_len)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            if do_negs:
                cur_new_negs = new_negs[i]
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    if do_negs:
                        new_negs_padded[i][-cur_len:] = cur_new_negs
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    if do_negs:
                        new_negs_padded[i][:cur_len] = cur_new_negs
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if negs is None:
            new_negs = None
        else:
            new_negs = new_negs_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # print(f'$$$$$$$$ negs posterior count = {simple_count(new_negs)} $$$$$$$$')

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_negs

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # leo added
        self.llm_tokenizer = tokenizer

        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
