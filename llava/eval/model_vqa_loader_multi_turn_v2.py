import argparse
import torch
import os
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

from llava.eval.eval_utils.perplexity_utils import get_full_prompt, DatasetForPerplexity_SEED
from llava.eval.eval_utils.dataset_utils import Dataset_SugarCrepe, shuffle_choices, Dataset_common_diff
import os.path as osp
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.dummy_image = None

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        qs_first_turn = line['first_turn_text']
        # if self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs  # DEFAULT_IMAGE_TOKEM = '<image>'
        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        # get_full_prompt():  add system prompt,  and  <image>,  USER:,  ASSISTANT:
        prompt_second_turn = get_full_prompt(conv_mode=args.conv_mode,qs= qs, mm_use_im_start_end= self.model_config.mm_use_im_start_end)[0] #  here prompt is a tuple
        prompt_second_turn = prompt_second_turn.replace(DEFAULT_IMAGE_TOKEN + '\n', '')
        prompt_first_turn = get_full_prompt(conv_mode=args.conv_mode,qs= qs_first_turn, mm_use_im_start_end= self.model_config.mm_use_im_start_end, two_imagse=True)[0]

        #  here if prompt is a tuple, the output input_ids is also a tuple ,   input_ids[0]
        input_ids_second_turn = tokenizer_image_token(prompt_second_turn, self.tokenizer, IMAGE_TOKEN_INDEX,  return_tensors='pt')  # IMAGE_TOKEN_INDEX = -200,
        input_ids_first_turn = tokenizer_image_token(prompt_first_turn, self.tokenizer, IMAGE_TOKEN_INDEX,  return_tensors='pt')

        # image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        # image_tensor = process_images([image], self.image_processor, self.model_config)[0]  # (3, 336, 336)

        try:
            if isinstance(image_file, list):
                image_file_list = image_file
                image_full_path_list = [osp.join(self.image_folder, image_file) for image_file in image_file_list]
                image_tensor = process_images(
                    [Image.open(image_full_path).convert('RGB') for image_full_path in image_full_path_list],
                    self.image_processor, self.model_config) # (2, 3, 336, 336)
            else:
                image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB') # (3, 336, 336)
                image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        except:
            image_tensor = self.dummy_image

        if self.dummy_image is None:
            import copy
            self.dummy_image = copy.deepcopy(image_tensor) * 0



        # return input_ids, image_tensor
        return input_ids_second_turn, prompt_second_turn, input_ids_first_turn, prompt_first_turn, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, args = None, annotation_file = None):
    assert batch_size == 1, "batch_size must be 1"
    if args.eval_dataset == 'SEED':
        if args.inference_mode == 'perplexity':
            dataset = DatasetForPerplexity_SEED(questions, image_folder, tokenizer, image_processor, model_config,  annotation_file=args.annotation_file,
                                                conv_mode=args.conv_mode, perplexity_prompt_version=args.perplexity_prompt_version)
        elif args.inference_mode == 'generate':
            dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
        else:
            raise NotImplementedError
    elif args.eval_dataset == 'SugarCrepe':
        dataset = Dataset_SugarCrepe(questions,  tokenizer, image_processor, model_config,
                    image_folder = image_folder,
                    eval_dataset=args.eval_dataset,
                    inference_mode=args.inference_mode,
                    perplexity_prompt_version=args.perplexity_prompt_version, conv_mode=args.conv_mode)
    elif args.eval_dataset == 'common_diff':
        # dataset = Dataset_common_diff(questions, tokenizer, image_processor, model_config,
        #                                 image_folder=image_folder,
        #                                 eval_dataset=args.eval_dataset,
        #                                 inference_mode=args.inference_mode,
        #                                 perplexity_prompt_version=args.perplexity_prompt_version, conv_mode=args.conv_mode)
        dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    elif args.eval_dataset in ['Ours_baseline', 'Ours_new']: # Irene's project,   SugarCrepe + ARO,  old and new negatives
        image_folder_dict = {'SugarCrepe': args.sugarcrepe_image_folder,
                             'ARO': args.aro_image_folder}
        dataset = Dataset_SugarCrepe(questions, tokenizer, image_processor, model_config,
                    image_folder_dict=image_folder_dict,
                    eval_dataset=args.eval_dataset,
                    inference_mode=args.inference_mode,
                    perplexity_prompt_version=args.perplexity_prompt_version, conv_mode=args.conv_mode, )
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

# <<<<<<< HEAD
    if args.question_file.endswith("jsonl"): # SEED-Bench,  no need to shuffle
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    elif args.question_file.endswith("json"):
        questions = json.load(open(os.path.expanduser(args.question_file), "r"))["questions"]
        if args.eval_dataset in  ['Ours_baseline', 'Ours_new']:
            for question_id, item in questions.items():
                if args.eval_dataset == 'Ours_baseline':
                    item['answers'] = [item['orig_pos'], item['orig_neg']]
                elif args.eval_dataset == 'Ours_new':
                    item['answers'] = [item['orig_pos'], item['new_neg']]
                else:
                    raise ValueError(f'Unknown eval_dataset: {args.eval_dataset}')
        if args.eval_dataset in ['SugarCrepe', 'Ours_baseline', 'Ours_new']:
            questions = shuffle_choices(questions)
            questions = list(questions.values())
    # get the chunk of questions for this process
    # data: question_id, question, answers, gt_answer, question_type_id, image_file,
    # question_typ_dict:  question_type to question_type_id dict
# =======
#     # model = model.bfloat16()

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
# >>>>>>> leonid
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, args=args) # by default, batch_size=1

    if args.inference_mode == 'perplexity':
# <<<<<<< HEAD
#         if args.eval_dataset == 'SugarCrepe':
#             answer_2options = ['A', 'B']
#             for (image_tensor, input_ids_a, labels_a, input_ids_b, labels_b), line in tqdm(zip(data_loader, questions),  total=len(questions)):
#                 idx = line["question_id"]
#                 gt_answer_idx = line['gt_answer_idx']  # the index of the true caption after shuffling
#                 image_tensor, input_ids_a, labels_a, input_ids_b, labels_b = (image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
#                                                                               input_ids_a.to(device='cuda', non_blocking=True), labels_a.to(device='cuda', non_blocking=True),
#                                                                               input_ids_b.to(device='cuda', non_blocking=True), labels_b.to(device='cuda', non_blocking=True))
#                 all_input_ids = [input_ids_a, input_ids_b]
#                 all_labels = [labels_a, labels_b]
#                 scores = []
#                 with torch.inference_mode():
#                     for input_ids, labels in zip(all_input_ids, all_labels):
#                         scores.append(  - model( input_ids=input_ids,images=image_tensor,labels=labels, return_dict=True)["loss"].to('cpu').numpy() )
#                 pred = scores.index(max(scores))
#                 pred_answer = answer_2options[pred]
#                 ans_file.write(json.dumps({"question_id": idx,
#                                            "text": pred_answer,
#                                            "gt_answer": answer_2options[gt_answer_idx], # # the letter of the true caption after shuffling
#                                            "answer_id": shortuuid.uuid(),
#                                            "model_id": model_name,
#                                            "metadata": {}}) + "\n")
#         else:
#             answer_4options = ['A', 'B', 'C', 'D']
#             for (image_tensor, input_ids_a, labels_a, input_ids_b,  labels_b, input_ids_c, labels_c, input_ids_d, labels_d), line in tqdm(zip(data_loader, questions), total=len(questions)):
#                 if image_tensor is None:
#                     continue
#                 idx = line["question_id"]
#                 question = data_loader.dataset.annotated_data_dict[str(idx)]['question']
#                 choices = [data_loader.dataset.annotated_data_dict[str(idx)]['choice_a'], data_loader.dataset.annotated_data_dict[str(idx)]['choice_b'],
#                            data_loader.dataset.annotated_data_dict[str(idx)]['choice_c'], data_loader.dataset.annotated_data_dict[str(idx)]['choice_d']]
#                 gt_answer = data_loader.dataset.annotated_data_dict[str(idx)]['answer']
#                 # gt_answer_idx = seed_answer_conversion( data_loader.dataset.annotated_data_dict[str(idx)]['answer'])
#                 image_tensor, input_ids_a, labels_a, input_ids_b, labels_b, input_ids_c, labels_c, input_ids_d, labels_d \
#                     = (image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
#                        input_ids_a.to(device='cuda', non_blocking=True), labels_a.to(device='cuda', non_blocking=True),
#                        input_ids_b.to(device='cuda', non_blocking=True), labels_b.to(device='cuda', non_blocking=True),
#                        input_ids_c.to(device='cuda', non_blocking=True), labels_c.to(device='cuda', non_blocking=True),
#                        input_ids_d.to(device='cuda', non_blocking=True), labels_d.to(device='cuda', non_blocking=True))
#                 all_input_ids = [input_ids_a, input_ids_b, input_ids_c, input_ids_d]
#                 all_labels = [labels_a, labels_b, labels_c, labels_d]
#                 scores = []
#                 with torch.inference_mode():
#                     for input_ids, labels in zip(all_input_ids, all_labels):
#                         scores.append(  - model( input_ids=input_ids,images=image_tensor,labels=labels, return_dict=True)["loss"].to('cpu').numpy() )
#                 pred = scores.index(max(scores))
#                 # hit = int(pred == gt_answer_idx)
#                 pred_answer = answer_4options[pred]
#                 ans_file.write(json.dumps({"question_id": idx,
#                                            # "prompt": question,
#                                            "question": question,
#                                             "choices": choices,
#                                             "gt_answer": gt_answer,
#                                            "text": pred_answer,
#                                            "answer_id": shortuuid.uuid(),
#                                            "model_id": model_name,
#                                            "metadata": {}}) + "\n")
# =======
        seed_answer_options = ['A', 'B', 'C', 'D']
        for (image_tensor, input_ids_a, labels_a, input_ids_b,
             labels_b, input_ids_c, labels_c, input_ids_d, labels_d), line in tqdm(zip(data_loader, questions), total=len(questions)):
            if image_tensor is None:
                continue
            idx = line["question_id"]
            question = data_loader.dataset.annotated_data_dict[str(idx)]['question']
            choices = [data_loader.dataset.annotated_data_dict[str(idx)]['choice_a'], data_loader.dataset.annotated_data_dict[str(idx)]['choice_b'],
                       data_loader.dataset.annotated_data_dict[str(idx)]['choice_c'], data_loader.dataset.annotated_data_dict[str(idx)]['choice_d']]
            gt_answer = data_loader.dataset.annotated_data_dict[str(idx)]['answer']
            # gt_answer_idx = seed_answer_conversion( data_loader.dataset.annotated_data_dict[str(idx)]['answer'])
            image_tensor, input_ids_a, labels_a, input_ids_b, labels_b, input_ids_c, labels_c, input_ids_d, labels_d \
                = (image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                   input_ids_a.to(device='cuda', non_blocking=True), labels_a.to(device='cuda', non_blocking=True),
                   input_ids_b.to(device='cuda', non_blocking=True), labels_b.to(device='cuda', non_blocking=True),
                   input_ids_c.to(device='cuda', non_blocking=True), labels_c.to(device='cuda', non_blocking=True),
                   input_ids_d.to(device='cuda', non_blocking=True), labels_d.to(device='cuda', non_blocking=True))
            all_input_ids = [input_ids_a, input_ids_b, input_ids_c, input_ids_d]
            all_labels = [labels_a, labels_b, labels_c, labels_d]
            scores = []
            with torch.inference_mode():
                for input_ids, labels in zip(all_input_ids, all_labels):
                    scores.append(  - model( input_ids=input_ids.long(),images=image_tensor,labels=labels, return_dict=True)["loss"].to('cpu').numpy() )
            pred = scores.index(max(scores))
            # hit = int(pred == gt_answer_idx)
            pred_answer = seed_answer_options[pred]
            ans_file.write(json.dumps({"question_id": idx,
                                       # "prompt": question,
                                       "question": question,
                                        "choices": choices,
                                        "gt_answer": gt_answer,
                                       "text": pred_answer,
                                       "answer_id": shortuuid.uuid(),
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
# >>>>>>> leonid
    elif args.inference_mode == 'generate':
        # if args.eval_dataset in ['SugarCrepe', 'Ours_baseline', 'Ours_new']:
        #     answer_2options = ['A', 'B']
        for data_tuple, line in tqdm(zip(data_loader, questions), total=len(questions)):
            # input_ids, image_tensor = data_tuple
            _, _, input_ids_first_turn, prompt_first_turn, image_tensor = data_tuple
            if image_tensor.dim() == 5:
                image_tensor = image_tensor.squeeze(0)
            if image_tensor is None:
                continue

            idx = line["question_id"]
            question_second_turn = line["text"]



            if isinstance(input_ids_first_turn, list):
                input_ids_first_turn = input_ids_first_turn[0]
            input_ids_first_turn = input_ids_first_turn.to(device='cuda', non_blocking=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids_first_turn, # (1, 83)  containing the -200
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True), # (1, 3, 336, 336)
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)  #  in generate mode, there is no labels provided

            input_token_len = input_ids_first_turn.shape[1]
            n_diff_input_output = (input_ids_first_turn != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()

            # prompt_second_turn

            second_turn_prompt = prompt_first_turn[0] + ' ' + outputs + '</s>USER: ' + question_second_turn + ' ASSISTANT:'
            input_ids_second_turn = tokenizer_image_token(second_turn_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            if isinstance(input_ids_second_turn, list):
                input_ids_second_turn = input_ids_second_turn[0]
            input_ids_second_turn = input_ids_second_turn.to(device='cuda', non_blocking=True)
            input_ids_second_turn  = input_ids_second_turn.reshape((1, -1))
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids_second_turn,  # (1, 83)  containing the -200
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),  # (1, 3, 336, 336)
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)
            input_token_len = input_ids_second_turn.shape[1]
            n_diff_input_output = (input_ids_second_turn != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs_ = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs_ = outputs_.strip()


            # # hotfix :-)
            # _outputs = outputs.split('</s>')
            # outputs = _outputs[0]

            ans_id = shortuuid.uuid()

            ans_file.write(json.dumps({"question_id": idx,
                                       "image": line["image"],
                                       "outputs_first_turn": outputs,
                                       "prompt": question_second_turn,
                                       "text": outputs_,
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}
                                       }) + "\n")

            # if args.eval_dataset in ['SugarCrepe', 'Ours_baseline', 'Ours_new']:
            #     ans_file.write(json.dumps({"question_id": idx,
            #                                "text": outputs, # outputs is the answer option  A or B
            #                                  "gt_answer": answer_2options[gt_answer_idx],
            #                                "answer_id": ans_id,
            #                                "model_id": model_name,
            #                                "metadata": {}
            #                                }) + "\n")
            # elif args.eval_dataset == 'common_diff':
            #     ans_file.write(json.dumps({"question_id": idx,
            #                                "image": line["image"],
            #                                "prompt": question_second_turn,
            #                                "text": outputs,
            #                                "answer_id": ans_id,
            #                                "model_id": model_name,
            #                                "metadata": {}
            #                                }) + "\n")
            # else: # SEED
            #     ans_file.write(json.dumps({"question_id": idx,
            #                                "prompt": question_second_turn,
            #                                "text": outputs,
            #                                "answer_id": ans_id,
            #                                "model_id": model_name,
            #                                "metadata": {}}) + "\n")
            # # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--sugarcrepe-image-folder", type=str, default="")
    parser.add_argument("--aro-image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--inference_mode", type=str, default='generate')
    parser.add_argument("--perplexity_prompt_version", type=str, default='v1')
    parser.add_argument("--eval_dataset", type=str, default='SEED')
    parser.add_argument("--annotation_file", type=str, default='/system/user/publicdata/LMM_benchmarks/SEED-Bench/SEED-Bench.json')

    args = parser.parse_args()

    # from cvar_pyutils.debugging_tools import set_remote_debugger
    # set_remote_debugger(None, 12345)

    eval_model(args)
