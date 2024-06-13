import argparse
import torch
import os
import os.path as osp
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import numpy as np

def get_full_prompt(conv_mode, qs, mm_use_im_start_end, two_imagse=False):
    """
    add system prompt,  and  <image>,  USER:,  ASSISTANT:
    e.g.

    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
    What is the color of the chair seen on the right side of the image? ASSISTANT:


    """
    if not DEFAULT_IMAGE_TOKEN in qs:
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            if two_imagse:
                qs = (f'Image 1: {DEFAULT_IMAGE_TOKEN}\n'
                      f'Image 2: {DEFAULT_IMAGE_TOKEN}\n') + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt   #  returned prompt is a tuple


choice_letters = ['A', 'B', 'C', 'D']


class DatasetForPerplexity_SEED(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config,  annotation_file = None,
                 perplexity_prompt_version = 'v1', conv_mode = None):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.dummy_image = None
        self.perplexity_prompt_version = perplexity_prompt_version
        self.conv_mode = conv_mode

        self.data = json.load(open(annotation_file))
        self.annotated_data_dict = dict()
        for item in self.data['questions']:
            self.annotated_data_dict[item['question_id']] = item

    def get_labels(self, instruction_w_answer, instruction_only, ignore_index, image_position =None, image_token_index=None):
        #  #  only the response tokens are not -100, the rest (system_promt, image placeholder, instructions) are -100
        labels = instruction_w_answer.clone()
        labels[: len(instruction_only)] = ignore_index
        # labels[image_position] = image_token_index
        return labels

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        question_w_options = line["text"] # question with options and answer template e.g. "How many towels are in the image? A. One B. Two C. Three D. Four Answer with the option's letter from the given choices directly."
        question_id = line["question_id"]
        question = self.annotated_data_dict[str(question_id)]['question'] # question only
        # answer texts only, no letter
        answer_texts = [self.annotated_data_dict[str(question_id)]['choice_a'], self.annotated_data_dict[str(question_id)]['choice_b'],
                   self.annotated_data_dict[str(question_id)]['choice_c'], self.annotated_data_dict[str(question_id)]['choice_d']]
        if self.perplexity_prompt_version == 'v1':
            # A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
            #     What is the color of the chair seen on the right side of the image? ASSISTANT:
            extended_instruction = question
            extended_responses = answer_texts

        elif self.perplexity_prompt_version == 'v2':
            # "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
            # What is the color of the chair seen on the right side of the image?\n
            # A. White\n B. Black\n C. Gray D. Brown\n
            # Answer with both the letter and text of the option from the given choices. ASSISTANT:"
            new_template = "Answer with both the letter and text of the option from the given choices."
            seg_list = question_w_options.split('\n')
            extended_instruction = '\n'.join(seg_list[:-1]) + '\n' + new_template
            extended_responses = [seg_list[1], seg_list[2], seg_list[3], seg_list[4]] #  A. White B. Black C. Gray D. Brown
        elif self.perplexity_prompt_version == 'v3':
            """
            A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
            What is the color of the chair seen on the right side of the image?\n
            A. White\n B. Black\n C. Gray\n D. Brown\n
            Answer with the option's letter from the given choices directly. ASSISTANT:
            """
            extended_instruction = question_w_options
            extended_responses = choice_letters
        elif self.perplexity_prompt_version == 'v4':
            """
            A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
            What is the color of the chair seen on the right side of the image?\n
            Answer in short words. ASSISTANT:
            """
            extended_instruction = question + '\n' + 'Answer in short words.'
            extended_responses = answer_texts
        elif self.perplexity_prompt_version == 'v5':
            """
            A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
            What is the color of the chair seen on the right side of the image?
            White
            Black
            Gray
            Brown
            Answer with one option from the given choices. ASSISTANT:+
            """
            extended_instruction = question + '\n' + '\n'.join(answer_texts) + '\n' + 'Answer with one option from the given choices.'
            extended_responses = answer_texts
        elif self.perplexity_prompt_version == 'v6':
            """
            A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
            What is the color of the chair seen on the right side of the image?
            White
            Black
            Gray
            Brown
            Answer in short words. ASSISTANT:+
            """
            extended_instruction = question + '\n' + '\n'.join(
                answer_texts) + '\n' + 'Answer in short words.'
            extended_responses = answer_texts

        full_instruction_prompt = get_full_prompt(conv_mode=self.conv_mode, qs=extended_instruction,
                                                  mm_use_im_start_end=self.model_config.mm_use_im_start_end)[0]  # here prompt is a tuple
        input_ids_instruction = tokenizer_image_token(full_instruction_prompt, self.tokenizer,
                                                      IMAGE_TOKEN_INDEX,  return_tensors='pt')  # IMAGE_TOKEN_INDEX = -200,
        # image_position = (input_ids_instruction == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].numpy()[0]

        # input_ids_a = tokenizer_image_token(full_instruction_w_answer_list[0], self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') # IMAGE_TOKEN_INDEX = -200,
        # input_ids_b =  tokenizer_image_token(full_instruction_w_answer_list[1], self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        # input_ids_c = tokenizer_image_token(full_instruction_w_answer_list[2], self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        # input_ids_d = tokenizer_image_token(full_instruction_w_answer_list[3], self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        input_ids_choice_a = self.tokenizer(text=extended_responses[0])['input_ids']
        input_ids_choice_b = self.tokenizer(text=extended_responses[1])['input_ids']
        input_ids_choice_c = self.tokenizer(text=extended_responses[2])['input_ids']
        input_ids_choice_d = self.tokenizer(text=extended_responses[3])['input_ids']

        # A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
        #     What is the color of the chair seen on the right side of the image? ASSISTANT: White
        input_ids_a = torch.cat((input_ids_instruction, torch.tensor(input_ids_choice_a[1:])))  # remove the start token of the choice
        input_ids_b = torch.cat((input_ids_instruction, torch.tensor(input_ids_choice_b[1:])))
        input_ids_c = torch.cat((input_ids_instruction, torch.tensor(input_ids_choice_c[1:])))
        input_ids_d = torch.cat((input_ids_instruction, torch.tensor(input_ids_choice_d[1:])))

        # Interesting, when I concatenate these two token segments,  I did not have to add white space after “ASSISTANT:”
        # But when I compute the token for the entire text sequence. I need to add white space after  “ASSISTANT:”

        # full_instruction_w_answer_list = [full_instruction_prompt + ' ' + extended_response for extended_response in extended_responses ]
        # input_ids_a__ = tokenizer_image_token(full_instruction_w_answer_list[0], self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        #  only the response tokens are not -100, the rest (system_promt, image placeholder, instructions) are -100
        labels_a = self.get_labels(instruction_w_answer=input_ids_a, instruction_only=input_ids_instruction,ignore_index=IGNORE_INDEX)
        labels_b = self.get_labels(instruction_w_answer=input_ids_b, instruction_only=input_ids_instruction, ignore_index=IGNORE_INDEX)
        labels_c = self.get_labels(instruction_w_answer=input_ids_c, instruction_only=input_ids_instruction,ignore_index=IGNORE_INDEX)
        labels_d = self.get_labels(instruction_w_answer=input_ids_d, instruction_only=input_ids_instruction,ignore_index=IGNORE_INDEX)

        try:
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB') # (1024, 705)
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        except:
            image_tensor = self.dummy_image

        if self.dummy_image is None:
            import copy
            self.dummy_image = copy.deepcopy(image_tensor) * 0

        return image_tensor, input_ids_a, labels_a, input_ids_b, labels_b, input_ids_c, labels_c, input_ids_d, labels_d
    def __len__(self):
        return len(self.questions)
