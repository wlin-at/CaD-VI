
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
from llava.eval.eval_utils.perplexity_utils import get_full_prompt
import torch

letters_2options = ['A. ', 'B. ']
answer_letters_2options = ['A', 'B']

def shuffle_choices(questions):  # before shuffling, the first answer is the correct one
    for question_id, item in questions.items():
        ixx = np.arange(len(item['answers']))
        np.random.shuffle(ixx)
        item['answers'] = np.array(item['answers'])[ixx]
        item['gt_answer_idx'] = np.where(ixx == 0)[0][0]
    return questions


class Dataset_common_diff(Dataset):
    def __init__(self, questions, tokenizer, image_processor, model_config,
                 image_folder=None,
                 eval_dataset=None,
                 image_folder_dict=None,
                 inference_mode='generate',
                 perplexity_prompt_version='v1', conv_mode=None):
        self.questions = questions
        self.image_folder = image_folder
        self.image_folder_dict = image_folder_dict
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.eval_dataset = eval_dataset
        self.perplexity_prompt_version = perplexity_prompt_version
        self.conv_mode = conv_mode
        self.inference_mode = inference_mode

    def get_labels(self, instruction_w_answer, instruction_only, ignore_index, image_position =None, image_token_index=None):
        #  #  only the response tokens are not -100, the rest (system_promt, image placeholder, instructions) are -100
        labels = instruction_w_answer.clone()
        labels[: len(instruction_only)] = ignore_index
        # labels[image_position] = image_token_index
        return labels
    def __getitem__(self, index):
        line = self.questions[index]
        image_file_list = line['image']
        image_full_path_list = [osp.join(self.image_folder, image_file) for image_file in image_file_list]
        image_tensor = process_images([Image.open(image_full_path).convert('RGB') for image_full_path in image_full_path_list],
                                           self.image_processor, self.model_config)
        # image_tensor = torch.stack(image_tensor_list, dim=0)
        question = line['text']
        if self.inference_mode == 'generate':
            full_instruction_prompt = get_full_prompt(conv_mode=self.conv_mode, qs=question,
                                                      mm_use_im_start_end=self.model_config.mm_use_im_start_end, two_imagse=True)[0]
            input_ids_instruction = tokenizer_image_token(full_instruction_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            return input_ids_instruction, image_tensor
        elif self.inference_mode == 'perplexity' and 'winoground' in self.image_folder.lower():
            """
            Image 1: <image>\n
            Image 2: <image>\n
            f'Caption 1: {cap1}\n'
            f'Caption 2: {cap2}'
            """
            # TODO  change caption 1 into the   caption text
            if False:
                extended_instruction1 = question + '\n' + 'Does Caption 1 suit Image 1? Answer with Yes or No.'
                extended_instruction2 = question + '\n' + 'Does Caption 1 suit Image 2? Answer with Yes or No.'
                extended_instruction3 = question + '\n' + 'Does Caption 2 suit Image 1? Answer with Yes or No.'
                extended_instruction4 = question + '\n' + 'Does Caption 2 suit Image 2? Answer with Yes or No.'
            else:
                items = question.split('\n')
                cap1 = items[-2].replace('Caption 1: ', '')
                cap2 = items[-1].replace('Caption 2: ', '')
                question = '\n'.join(items[:-2])
                question = question + '\n' + f'There are two captions.\n{cap1}\n{cap2}'
                extended_instruction1 = question + '\n' + f'Does the caption "{cap1}" suit Image 1? Answer with Yes or No.'
                extended_instruction2 = question + '\n' + f'Does the caption "{cap1}" suit Image 2? Answer with Yes or No.'
                extended_instruction3 = question + '\n' + f'Does the caption "{cap2}" suit Image 1? Answer with Yes or No.'
                extended_instruction4 = question + '\n' + f'Does the caption "{cap2}" suit Image 2? Answer with Yes or No.'
            extended_response = 'Yes'
            full_instruct_prompt1 = get_full_prompt(conv_mode=self.conv_mode, qs=extended_instruction1,mm_use_im_start_end=self.model_config.mm_use_im_start_end)[0]
            full_instruct_prompt2 = get_full_prompt(conv_mode=self.conv_mode, qs=extended_instruction2,mm_use_im_start_end=self.model_config.mm_use_im_start_end)[0]
            full_instruct_prompt3 = get_full_prompt(conv_mode=self.conv_mode, qs=extended_instruction3,mm_use_im_start_end=self.model_config.mm_use_im_start_end)[0]
            full_instruct_prompt4 = get_full_prompt(conv_mode=self.conv_mode, qs=extended_instruction4,mm_use_im_start_end=self.model_config.mm_use_im_start_end)[0]
            input_ids_instruction1 = tokenizer_image_token(full_instruct_prompt1, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids_instruction2 = tokenizer_image_token(full_instruct_prompt2, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids_instruction3 = tokenizer_image_token(full_instruct_prompt3, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids_instruction4 = tokenizer_image_token(full_instruct_prompt4, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

            input_ids_answer = self.tokenizer(text=extended_response)['input_ids']
            input_ids_a = torch.cat((input_ids_instruction1, torch.tensor(input_ids_answer[1:]))) # remove the start token of the choice
            input_ids_b = torch.cat((input_ids_instruction2, torch.tensor(input_ids_answer[1:])))
            input_ids_c = torch.cat((input_ids_instruction3, torch.tensor(input_ids_answer[1:])))
            input_ids_d = torch.cat((input_ids_instruction4, torch.tensor(input_ids_answer[1:])))
            labels_a = self.get_labels(instruction_w_answer=input_ids_a, instruction_only=input_ids_instruction1, ignore_index=IGNORE_INDEX)
            labels_b = self.get_labels(instruction_w_answer=input_ids_b, instruction_only=input_ids_instruction2, ignore_index=IGNORE_INDEX)
            labels_c = self.get_labels(instruction_w_answer=input_ids_c, instruction_only=input_ids_instruction3, ignore_index=IGNORE_INDEX)
            labels_d = self.get_labels(instruction_w_answer=input_ids_d, instruction_only=input_ids_instruction4, ignore_index=IGNORE_INDEX)
            return image_tensor, input_ids_a, labels_a, input_ids_b, labels_b, input_ids_c, labels_c, input_ids_d, labels_d
        elif self.inference_mode == 'perplexity' and 'SEED-Bench2' in self.image_folder:
            answer_texts = [line['choice_a'], line['choice_b'], line['choice_c'], line['choice_d']]
            extended_instruction = question
            full_instruction_prompt = get_full_prompt(conv_mode=self.conv_mode, qs=extended_instruction, mm_use_im_start_end=self.model_config.mm_use_im_start_end)[0]
            input_ids_instruction = tokenizer_image_token(full_instruction_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids_choice_list = [self.tokenizer(text=answer_text)['input_ids'] for answer_text in answer_texts]
            input_ids_list = [torch.cat((input_ids_instruction, torch.tensor(input_ids_choice[1:]))) for input_ids_choice in input_ids_choice_list]# remove the start token of the choice
            labels_list = [self.get_labels(instruction_w_answer=input_ids, instruction_only=input_ids_instruction, ignore_index=IGNORE_INDEX) for input_ids in input_ids_list]
            return image_tensor, input_ids_list[0], labels_list[0], input_ids_list[1], labels_list[1], input_ids_list[2], labels_list[2], input_ids_list[3], labels_list[3]
    def __len__(self):
        return len(self.questions)
class Dataset_SugarCrepe(Dataset):
    def __init__(self, questions,  tokenizer, image_processor, model_config,
                 image_folder = None,
                 eval_dataset = None,
                 image_folder_dict = None,
                 inference_mode = None,
                 eval_paritions = ['add_obj', 'add_att', 'replace_obj', 'replace_att', 'replace_rel', 'swap_obj', 'swap_att'],
                 perplexity_prompt_version = 'v1', conv_mode = None):
        self.questions = questions
        self.image_folder = image_folder #  path of the COCO images, /system/user/publicdata/LMM_benchmarks/SugarCrepe/val2017
        self.image_folder_dict = image_folder_dict
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.dummy_image = None
        self.eval_dataset = eval_dataset
        # self.eval_paritions = eval_paritions
        self.perplexity_prompt_version = perplexity_prompt_version
        self.conv_mode = conv_mode
        self.inference_mode = inference_mode

        t= 1

    def get_labels(self, instruction_w_answer, instruction_only, ignore_index):
        #  #  only the response tokens are not -100, the rest (system_promt, image placeholder, instructions) are -100
        labels = instruction_w_answer.clone()
        labels[: len(instruction_only)] = ignore_index
        # labels[image_position] = image_token_index
        return labels

    def __getitem__(self, index):
        line = self.questions[index]

        shuffled_opt_txt = np.array(line['answers'])  # this list is already shuffled
        if self.eval_dataset == 'SugarCrepe':
            image_file = line['image_file']
            image_full_path = osp.join(self.image_folder, image_file)
        elif self.eval_dataset in ['Ours_baseline', 'Ours_new']:
            image_file = line['image_file']
            image_full_path = osp.join(self.image_folder_dict[line['image_source']], image_file)
        else:
            raise ValueError(f'Unknown eval_dataset: {self.eval_dataset}')
        image = Image.open(image_full_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        if self.inference_mode == 'generate':
            """
            Which is a suitable caption for this image?
        A. A cat and a dog napping together under a blanket on the couch.
        B. A cat and dog napping together on the couch.
        Answer with the option's letter from the given choices directly.
            """
            # line = self.questions[index]
            # question_id = line['question_id']
            # opt_txt = np.array(line['answers']) # [true caption, false caption]
            # ixx = np.arange(len(opt_txt))
            # np.random.shuffle(ixx)
            # gt = np.where(ixx == 0)[0][0]   # get the index of the true caption after shuffling
            # shuffled_opt_txt = opt_txt[ixx]


            # question_id = line['question_id']

            # gt = line['gt_answer_idx'] # get the index of the true caption after shuffling
            extended_instruction = (("Which is a suitable caption for this image?\n"
                                     + "\n".join([letters_2options[i] + shuffled_opt_txt[i] for i in range(len(shuffled_opt_txt))]))
                            + "\n" + "Answer with the option's letter from the given choices directly.")
            full_instruction_prompt = get_full_prompt(conv_mode = self.conv_mode, qs=extended_instruction, mm_use_im_start_end=self.model_config.mm_use_im_start_end)[0]
            input_ids_instruction = tokenizer_image_token(full_instruction_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            return input_ids_instruction, image_tensor


        elif self.inference_mode == 'perplexity':
            if self.perplexity_prompt_version == 'v3':
                """
                Which is a suitable caption for this image?\n
            A. A cat and a dog napping together under a blanket on the couch.\n
            B. A cat and dog napping together on the couch.\n
            Answer with the option's letter from the given choices directly.
                """
                extended_instruction =  (("Which is a suitable caption for this image?\n"
                                          + "\n".join(
                    [letters_2options[i] + shuffled_opt_txt[i] for i in range(len(shuffled_opt_txt))]))
                                    + "\n" + "Answer with the option's letter from the given choices directly.")
                extended_responses = answer_letters_2options
            full_instruction_prompt = get_full_prompt(conv_mode=self.conv_mode, qs=extended_instruction,
                         mm_use_im_start_end=self.model_config.mm_use_im_start_end)[0]  # here prompt is a tuple
            input_ids_instruction = tokenizer_image_token(full_instruction_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')  # IMAGE_TOKEN_INDEX = -200,
            input_ids_choice_a = self.tokenizer(text=extended_responses[0])['input_ids']
            input_ids_choice_b = self.tokenizer(text=extended_responses[1])['input_ids']

            input_ids_a = torch.cat(
                (input_ids_instruction, torch.tensor(input_ids_choice_a[1:])))  # remove the start token of the choice
            input_ids_b = torch.cat((input_ids_instruction, torch.tensor(input_ids_choice_b[1:])))

            labels_a = self.get_labels(instruction_w_answer=input_ids_a, instruction_only=input_ids_instruction,
                                       ignore_index=IGNORE_INDEX)
            labels_b = self.get_labels(instruction_w_answer=input_ids_b, instruction_only=input_ids_instruction,
                                       ignore_index=IGNORE_INDEX)
            # image = Image.open(osp.join(self.image_folder, image_file)).convert('RGB') # (3, 336, 336)
            # image_tensor = process_images([image], self.image_processor, self.model_config)[0]

            return image_tensor, input_ids_a, labels_a, input_ids_b, labels_b
    def __len__(self):
        return len(self.questions)
