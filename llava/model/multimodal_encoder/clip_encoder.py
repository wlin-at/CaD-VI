import torch
import torch.nn as nn

from typing import Any, Optional, Tuple, Union
import types

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoModelForCausalLM, AutoTokenizer

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):  #  modified by leonid
        super().__init__()

        self.is_loaded = False
        #  hacky way to pass the information that the vis encoder will receive a prefix into its attention layer 7 from the hidden state number 40 of the decoder LLM :-)
        _vision_tower = vision_tower.split(':') # 'openai/clip-vit-large-patch14-336:base_llm#[[40],5120];[[7],1024]'
        if len(_vision_tower) > 1:
            _latt_params = _vision_tower[1].split('#')
            self.latt_model_name = _latt_params[0] # 'base_llm'
            self.latt_mapping = [eval(x) for x in _latt_params[1].split(';')] # [[[40], 5120], [[7], 1024]]

            latt_proj = [nn.Linear(self.latt_mapping[0][1], self.latt_mapping[1][1]) for _ in self.latt_mapping[0][0]]
            self.latt_proj = nn.ModuleList(latt_proj)
        else:
            self.latt_model_name = None
        vision_tower = _vision_tower[0]

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        if self.latt_model_name is not None:
            # self.vision_tower = LattCLIPVisionModel.from_pretrained(self.vision_tower_name)

            if self.latt_model_name != 'base_llm':
                self.latt_model = AutoModelForCausalLM.from_pretrained(self.latt_model_name, trust_remote_code=True)
                self.latt_tokenizer = AutoTokenizer.from_pretrained(self.latt_model_name, trust_remote_code=True)
                self.latt_model.requires_grad_(False)
            else:
                self.latt_model = None
                self.latt_tokenizer = None

        # else:
        #     self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images, texts=None, llm=None, tokenizer=None):
        if self.latt_model_name is not None:
            if self.latt_model is not None:
                llm = self.latt_model
                tokenizer = self.latt_tokenizer
            texts = [' '.join(x) for x in texts]
            texts_tok = tokenizer(texts, return_tensors="pt", return_attention_mask=True)
            with torch.no_grad():
                res = llm(input_ids=texts_tok['input_ids'].cuda(), attention_mask=texts_tok['attention_mask'].cuda(), output_hidden_states=True)
                res = [res.hidden_states[ix].detach() for ix in self.latt_mapping[0][0]]
                for rr in res:
                    rr.requires_grad = True
                # res.hidden_states[40].shape = torch.Size([1, 14, 5120])

            #transform to be used by the vis encoder
            latt = [(ix, self.latt_proj[ixx](x)) for ixx, (ix, x) in enumerate(zip(self.latt_mapping[1][0], res))]
            print(torch.abs(self.latt_proj[0].weight).sum().item())

            for _latt in latt:
                latt_input = _latt[1]

                def latt_self_attn_forward(
                    self,
                    hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    causal_attention_mask: Optional[torch.Tensor] = None,
                    output_attentions: Optional[bool] = False,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
                    """Input shape: Batch x Time x Channel"""

                    # print('Someone called me')
                    # print(f'I see latt, its shape is {latt_input.shape}')

                    bsz, tgt_len, embed_dim = hidden_states.size()

                    # get query proj
                    query_states = self.q_proj(hidden_states) * self.scale
                    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
                    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

                    # concat latt
                    key_states_latt = self._shape(self.k_proj(latt_input), -1, bsz)
                    value_states_latt = self._shape(self.v_proj(latt_input), -1, bsz)
                    key_states = torch.cat((key_states_latt,key_states), dim=2)
                    value_states = torch.cat((value_states_latt, value_states), dim=2)

                    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
                    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
                    key_states = key_states.view(*proj_shape)
                    value_states = value_states.view(*proj_shape)

                    src_len = key_states.size(1)
                    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

                    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
                        raise ValueError(
                            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                            f" {attn_weights.size()}"
                        )

                    # apply the causal_attention_mask first
                    if causal_attention_mask is not None:
                        if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                            raise ValueError(
                                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                                f" {causal_attention_mask.size()}"
                            )
                        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
                        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

                    if attention_mask is not None:
                        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                            raise ValueError(
                                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                            )
                        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
                        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

                    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

                    if output_attentions:
                        # this operation is a bit akward, but it's required to
                        # make sure that attn_weights keeps its gradient.
                        # In order to do so, attn_weights have to reshaped
                        # twice and have to be reused in the following
                        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
                    else:
                        attn_weights_reshaped = None

                    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

                    attn_output = torch.bmm(attn_probs, value_states)

                    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
                        raise ValueError(
                            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                            f" {attn_output.size()}"
                        )

                    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
                    attn_output = attn_output.transpose(1, 2)
                    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

                    attn_output = self.out_proj(attn_output)

                    return attn_output, attn_weights_reshaped

                monkey_inst = self.vision_tower.vision_model.encoder.layers[_latt[0]].self_attn
                monkey_inst.forward = types.MethodType(latt_self_attn_forward, monkey_inst) #, transformers.models.clip.modeling_clip.CLIPAttention)

        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
