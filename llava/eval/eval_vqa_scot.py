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

        # if self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs  # DEFAULT_IMAGE_TOKEM = '<image>'
        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        # get_full_prompt():  add system prompt,  and  <image>,  USER:,  ASSISTANT:
        prompt = get_full_prompt(conv_mode=args.conv_mode,qs= qs, mm_use_im_start_end= self.model_config.mm_use_im_start_end) #  here prompt is a tuple
        prompt_scot = get_full_prompt(conv_mode=args.conv_mode,qs= 'Densely annotate the image.', mm_use_im_start_end= self.model_config.mm_use_im_start_end)

        # image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        # image_tensor = process_images([image], self.image_processor, self.model_config)[0]  # (3, 336, 336)

        if self.dummy_image is None:
            import numpy as np
            zrimg = np.zeros((3, 336, 336))
            self.dummy_image = Image.fromarray(zrimg.astype('uint8'), 'RGB')

        try:
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB') # (3, 336, 336)
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        except:
            image = self.dummy_image
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        #  here if prompt is a tuple, the output input_ids is also a tuple ,   input_ids[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') # IMAGE_TOKEN_INDEX = -200,
        input_ids_scot = tokenizer_image_token(prompt_scot, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return (input_ids, prompt, input_ids_scot, prompt_scot), image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, args = None, annotation_file = None):
    assert batch_size == 1, "batch_size must be 1"
    if args.inference_mode == 'perplexity':
        dataset = DatasetForPerplexity_SEED(questions, image_folder, tokenizer, image_processor, model_config, eval_dataset=args.eval_dataset, annotation_file=args.annotation_file,
                                            conv_mode=args.conv_mode, perplexity_prompt_version=args.perplexity_prompt_version)
    elif args.inference_mode == 'generate':
        dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    else:
        raise NotImplementedError
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    if args.model_name is None:
        model_name = get_model_name_from_path(model_path)
    else:
        model_name = args.model_name
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, args=args) # by default, batch_size=1

    if args.inference_mode == 'perplexity':
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
                    scores.append(  - model( input_ids=input_ids,images=image_tensor,labels=labels, return_dict=True)["loss"].to('cpu').numpy() )
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
    elif args.inference_mode == 'generate':
        for ((input_ids, prompt, input_ids_scot, prompt_scot), image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
            if image_tensor is None:
                continue

            idx = line["question_id"]
            cur_prompt = line["text"] #  question with options

            if isinstance(input_ids_scot, list):
                input_ids_scot = input_ids_scot[0]
            input_ids_scot = input_ids_scot.to(device='cuda', non_blocking=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids_scot, # (1, 83)  containing the -200
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True), # (1, 3, 336, 336)
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=3000, #input_ids_scot
                    use_cache=True)  #  in generate mode, there is no labels provided

            input_token_len = input_ids_scot.shape[1]
            n_diff_input_output = (input_ids_scot != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()

            post_scot_prompt = prompt_scot[0][0] + outputs + ' USER: ' + prompt[0][0].split('<image>\n')[1]
            input_ids = tokenizer_image_token(post_scot_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

            if isinstance(input_ids, list):
                input_ids = input_ids[0]
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            input_ids = input_ids.reshape((1,-1))

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids, # (1, 83)  containing the -200
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True), # (1, 3, 336, 336)
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens, #input_ids_scot
                    use_cache=True)  #  in generate mode, there is no labels provided

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs_ = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs_ = outputs_.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": cur_prompt,
                                       "text": outputs_,
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
            # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
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

    from cvar_pyutils.debugging_tools import set_remote_debugger
    set_remote_debugger(None, 12345)

    eval_model(args)
