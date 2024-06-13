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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM #, LlamaConfig, LlamaModel, LlamaForCausalLM
from ibm_models import AttentionImplementation, GPTMegatronConfig, GPTMegatronModel, GPTMegatronForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaGraniteConfig(GPTMegatronConfig):
    model_type = "llava_granite"
    # tie_word_embeddings = False


class LlavaGraniteModel(LlavaMetaModel, GPTMegatronModel):
    config_class = LlavaGraniteConfig

    def __init__(self, config: GPTMegatronConfig):
        super(LlavaGraniteModel, self).__init__(config)

    def embed_tokens(self, x):
        return self.wte(x)

from llava.mm_utils import safe_dec
import sys

import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class LlavaGraniteForCausalLM(GPTMegatronForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaGraniteConfig

    def __init__(self, config):
        # config.tie_word_embeddings = False
        super(GPTMegatronForCausalLM, self).__init__(config)
        self.transformer = LlavaGraniteModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer

    def get_input_embeddings(self):
        return self.transformer.wte

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        negs: Optional[list] = None,
        token_type_ids: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        from llava.train.train import custom_args

        _orig_input_ids = input_ids  # system prompt token, -200, instruction token, response token

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                negs
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                negs
            )

        # leo changed     -    negs are not used here
        res = super().forward(
            input_ids=input_ids, # None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,  # (1, 658, 4096)
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        if (custom_args is not None) and custom_args.print_sample and (torch.distributed.get_rank() == 0):
            b, l, t = res.logits.shape
            if b > 0:
                rep_str = ''
                ib = 0
                for il in range(l):
                    rep_str += safe_dec(self.llm_tokenizer, labels[ib, il].item())
                print(rep_str)

        # if (torch.distributed.get_rank() == 0):
        #     print(f'$$$$ custom_args={custom_args}')

        # compute additional loss
        # print('### here')
        # print(f'### res is {type(res)}')
        # print(res)
        if (negs is not None) and (custom_args.loss_type in ['neg', 'focal', 'neg_focal', 'rand_focal']):
            # print('### negs found')
            logits = res.logits   #  (bz, max_len, 32000)
            # print(logits.shape)
            # print(len(negs[0]))
            # print(labels.shape)

            b, l, t = logits.shape
            cnt = 0
            cont_loss = 0

            reported = False
            num_top = custom_args.num_top_focal
            print_masked = True # to support False, we need to process the input_ids by adding the equivalent amount of tokens to what is added for the images

            for ib in range(b):
                if custom_args.loss_type == 'rand_focal':
                    negs[ib] = [[] for _ in range(l)]
                    non_masked = torch.nonzero(labels[ib, :] > 0).reshape((-1,)).numpy(force=True).tolist()
                    sample_ix = np.random.choice(non_masked, min(custom_args.num_rand_neg, len(non_masked)), replace=False)
                    for ix in sample_ix:
                        negs[ib][ix] = [1000]
                ln = min(len(negs[ib]), l)
                # assert (ln <= l)
                rep_str = ''
                neg_encountered = False
                for il in range(ln):  #   ib is batch id,  il is length id
                    n = negs[ib][il]
                    if print_masked:
                        rep_str += safe_dec(self.llm_tokenizer, labels[ib, il].item()) # labels are the masked version of input_ids ,   we suppose that -100 decodes to  @
                    else:
                        rep_str += safe_dec(self.llm_tokenizer, _orig_input_ids[ib, il].item())
                    if len(n) > 0:
                        cnt += 1
                        neg_encountered = True


                        # rep_str += f'{bcolors.WARNING}('
                        # rep_str += ', '.join([safe_dec(self.llm_tokenizer, x) for x in n])
                        # rep_str += f'){bcolors.ENDC}'
                        # #  logits (bz, max_len, 32000)
                        # n = [labels[ib, il].item()] + n  #  append the positive token into the beginning of the list, this becomes a list of a postive and several negatives
                        # s = logits[ib, il - 1, n] # so, the previous token actually predicts this one -  extracts the prediction logits of the previous token on the positive and negative tokens
                        # t = torch.topk(logits[ib, il - 1, :], num_top).indices  # extract the  top-3 largest logits (out of the 32000 scores), not used in the loss

                        t = torch.topk(logits[ib, il - 1, :], num_top).indices


                        if custom_args.loss_type == 'neg':
                            n = [labels[ib, il].item()] + n
                            s = logits[ib, il - 1, n]  # so, the previous token actually predicts this one
                            s = s[None, :]
                            cont_loss += torch.nn.functional.cross_entropy(s, torch.tensor([0]).to(logits.device))
                        elif (custom_args.loss_type == 'focal') or (custom_args.loss_type == 'rand_focal'):
                            r = [labels[ib, il].item()]
                            f = [x.item() for x in t if x.item() not in r]
                            n = [labels[ib, il].item()] + f
                            s = logits[ib, il - 1, n]  # so, the previous token actually predicts this one
                            s = s[None, :]
                            cont_loss += torch.nn.functional.cross_entropy(s, torch.tensor([0]).to(logits.device))
                        elif custom_args.loss_type == 'neg_focal':
                            r = [labels[ib, il].item()] + n
                            f = [x.item() for x in t if x.item() not in r]
                            n = [labels[ib, il].item()] + n + f
                            s = logits[ib, il - 1, n]  # so, the previous token actually predicts this one
                            s = s[None, :]
                            cont_loss += torch.nn.functional.cross_entropy(s, torch.tensor([0]).to(logits.device))

                        rep_str += f'{bcolors.WARNING}('
                        rep_str += ', '.join([safe_dec(self.llm_tokenizer, x) for x in n[1:]])
                        rep_str += f'){bcolors.ENDC}'

                        rep_str += f'{bcolors.OKGREEN}['
                        rep_str += ', '.join([safe_dec(self.llm_tokenizer, x.item()) for x in t])
                        rep_str += f']{bcolors.ENDC}'


                        # s = s[None, :]
                        # cont_loss += torch.nn.functional.cross_entropy(s, torch.tensor([0]).to(logits.device))  #  the ground truth label is 0, the positive token is the first one in the list

                if not reported and neg_encountered and (torch.distributed.get_rank() == 0):
                    reported = True
                    # print('%%%%%%%%%%')
                    print(rep_str)
                    # print('%%%%%%%%%%')

                    # dbg_inp, dbg_labs = '', ''
                    # for ii in range(labels.shape[1]):
                    #     dbg_labs += safe_dec(self.llm_tokenizer, labels[ib, ii].item())
                    # # for ii in range(input_ids.shape[1]):
                    # #     dbg_inp += safe_dec(self.llm_tokenizer, input_ids[ib, ii].item())
                    # if torch.distributed.get_rank() == 0:
                    #     # print('####################')
                    #     # print(dbg_inp)
                    #     print('####################')
                    #     print(dbg_labs)
                    #     print('####################')
                # sys.exit(0)

            if cnt > 0:
                cont_loss = cont_loss / cnt
            print(cont_loss)
            res['loss'] += cont_loss

        return res

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("llava_granite", LlavaGraniteConfig)
AutoModelForCausalLM.register(LlavaGraniteConfig, LlavaGraniteForCausalLM)
