import sys

from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def crop_two_square(pil_img):
    width, height = pil_img.size
    if width == height:
        return pil_img, pil_img.copy()
    elif width > height:
        # leftmost and rightmost images
        return pil_img.crop((0, 0, height, height)), pil_img.crop((width - height, 0, width, height))
        # return pil_img.crop(((height - width) // 2, 0, width + (height - width) // 2, width))
    else:
        # topmost and bottommost images
        return pil_img.crop((0, 0, width, width)), pil_img.crop((0, height - width, width, height))
        # return pil_img.crop((0, (width - height) // 2, height, height + (width - height) // 2))

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


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None) # pad
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean)) # expand to square, not crop
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] # tensor (3,336,336)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

# added by leo
def tokenizer_image_token_with_negs(tp, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):

    # leo - sanity
    # print(tp[0])
    # print(''.join(tp[1]))
    # print(''.join(tp[1]) == tp[0])
    # print(tp[1])
    # print(tp[2])
    # print((len(tp[1]), len(tp[2])))
    # if torch.distributed.get_rank()==0:
    #     print(list(zip(tp[1],tp[2])))
    # sys.exit(0)

    pos = tp[1]
    negs = tp[2]
    input_ids, input_ids_neg = [], []
    for (p, n) in zip(pos, negs):
        prompt = p
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep): # leo thinks this is a complicated way to add a sep between all consecutive lements of a list
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        offset = 0
        _input_ids = []
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            if len(input_ids) == 0:
                input_ids.append(prompt_chunks[0][0])
                input_ids_neg.append([])

        # if len(prompt_chunks) > 1:
        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            _input_ids.extend(x[offset:])
        # elif len(prompt_chunks) > 0:
        #     cln = prompt_chunks[0][offset:]
        #     _input_ids.extend(cln)

        input_ids.extend(_input_ids)
        _input_ids_neg = [[] for _ in _input_ids]
        if '<image>' not in prompt:
            # leo: assume that in pieces having negs we do not have any <image> tokens
            for neg in n:
                neg_tok = tokenizer(neg).input_ids[offset:]
                _input_ids_neg[0].append(neg_tok[0])
        input_ids_neg.extend(_input_ids_neg)

    # print(input_ids)
    # print(input_ids_neg)
    # print(len(input_ids))
    # print(len(input_ids_neg))

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long), input_ids_neg
        raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return input_ids, input_ids_neg

def safe_dec(t, x):
    try:
        return t.decode([x])
    except:
        if x == -100:
            return '@'
        else:
            return f'*{x}*'

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    #leo was here
    input_ids_neg = None
    if isinstance(prompt, tuple):  #  There is a debug, after modification by Leo in conversation.py, Conversation.get_prompt() always returns a tuple
        input_ids = tokenizer_image_token(prompt[0], tokenizer, image_token_index=image_token_index, return_tensors=None) # replace <image> with -200,  for the first chunk, keep the start token; for the rest chunks, remove the start token

        # slow and inefficient
        input_ids_neg = [[] for _ in input_ids]
        pos = prompt[1]
        negs = prompt[2]
        for ixx, neg in enumerate(negs):
            for n in neg:
                ref_input_ids = tokenizer_image_token(''.join(pos[:ixx] + [n]), tokenizer, image_token_index=image_token_index, return_tensors=None)
                dd = [ix for ix, (a, b) in enumerate(zip(input_ids, ref_input_ids)) if a != b]
                if len(dd) > 0:
                    input_ids_neg[dd[0]].append(ref_input_ids[dd[0]])
    else:
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0]) # the first token ( 1 ) is the start token,   for the first chunk, keep the start token; for the rest chunks, remove the start token

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        else:
            raise ValueError(f'Unsupported tensor type: {return_tensors}')

    if input_ids_neg is None:
        return input_ids
    else:
        return input_ids, input_ids_neg


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
