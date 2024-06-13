import os
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dataset", type=str, default='SEED')
    parser.add_argument("--annotation-file", type=str, default='/system/user/publicdata/LMM_benchmarks/SEED-Bench2/seedbench2_3_paritions_593.jsonl')
    parser.add_argument("--question-file", type=str, default='/system/user/publicdata/LMM_benchmarks/SEED-Bench2/seedbench2_3_paritions_593.jsonl')
    parser.add_argument("--result-file", type=str, default='/system/user/publicdata/LMM_benchmarks/SEED-Bench2/answers_generate/llava-v1.5-13b_lora_lr1e-4_943k_bz24x4_commondiff_v1_4bit_w_binary_image_select_NO_SHUFFLE/merge.jsonl')
    parser.add_argument("--result-upload-file", type=str, default=None)
    return parser.parse_args()


def eval_single(result_file, eval_dataset=None, eval_only_type=None, annotation_file = None): #  question_id is used to index the results
    results = {}
    for line in open(result_file):
        row = json.loads(line)
        results[row['question_id']] = row
    #  results is dictionary of dictionary,  101669: {'question_id': 101669, 'text': 'xx', 'answer_id': 'xxxx', 'model_id': 'xxxx', 'metadata': {}}
    type_counts = {}
    correct_counts = {}
    if 'seedbench2' in annotation_file:
        for type_id, category_name in ques_type_id_to_name.items():
            correct_counts.update({category_name: 0})
    else:
        for type_id in ques_type_id_to_name.keys():
            correct_counts.update({type_id: 0})
    # data has two components:
    #  'questions': 19242 items,
    #       each cotaining question_type_id, question_id, question, data_type, data_id, choice_a, choice_b, choice_c, choice_d, answer
    #  'question_type':  question_type to question_type_id dict

    if eval_dataset in ['SugarCrepe', 'Ours_baseline', 'Ours_new']:
        # # questions_list = list(data['questions'].values())
        # questions_list = list(results.values())
        # for question_data in questions_list:
        #     question_type = question_data['question_type_id']
        #     type_counts[question_type] = type_counts.get(question_type, 0) + 1
        for question_id, row in results.items():
            question_type = data['questions'][question_id]['question_type_id']
            type_counts[question_type] = type_counts.get(question_type, 0) + 1
            if row['text'] == row['gt_answer']:
                correct_counts[question_type] = correct_counts.get(question_type, 0) + 1

    else:
        questions_list = data['questions']

        for question_data in questions_list:
            if eval_only_type is not None and question_data['data_type'] != eval_only_type: continue
            if 'seedbench2' in annotation_file:
                question_type = question_data['category']
            else:
                question_type = question_data['question_type_id']
            type_counts[question_type] = type_counts.get(question_type, 0) + 1
            if 'seedbench2' in annotation_file:
                question_id = question_data['question_id']
            else:
                try:
                    question_id = int(question_data['question_id'])
                except:
                    question_id = question_data['question_id']
                if question_id not in results:
                    correct_counts[question_type] = correct_counts.get(question_type, 0)
                    continue
            row = results[question_id]
            if row['text'] == question_data['answer']:
                correct_counts[question_type] = correct_counts.get(question_type, 0) + 1

    total_count = 0
    total_correct = 0
    for question_type in sorted(type_counts.keys()):
        accuracy = correct_counts[question_type] / type_counts[question_type] * 100
        if eval_only_type is None:
            if 'seedbench2' in annotation_file:
                print(f"{question_type}: {accuracy:.2f}%")
            else:
                print(f"{ques_type_id_to_name[question_type]}: {accuracy:.2f}%")

        total_count += type_counts[question_type]
        total_correct += correct_counts[question_type]

    total_accuracy = total_correct / total_count * 100
    if eval_only_type is None:
        if eval_dataset in ['SugarCrepe', 'Ours_baseline', 'Ours_new']:
            total_str = 'Total_accuracy'
        else:
            if 'seedbench2' in annotation_file:
                total_str = 'Total_accuracy'
            else:
                total_str = 'Total accuracy'
        print(f"{total_str}: {total_accuracy:.2f}%")
    else:
        print(f"{eval_only_type} accuracy: {total_accuracy:.2f}%")

    return results

if __name__ == "__main__":
    args = get_args()
    if args.eval_dataset in ['SugarCrepe', 'Ours_baseline', 'Ours_new']:
        data = json.load(open(args.question_file))
    else:

        if args.annotation_file.endswith('.jsonl'):
            category_name_list = []
            data = {'questions': []}
            for line in open(args.annotation_file):
                sample = json.loads(line)
                if sample['category'] not in category_name_list:
                    category_name_list.append(sample['category'])
                data['questions'].append(sample)
            ques_type_id_to_name = {i: name for i, name in enumerate(category_name_list)}
        elif args.annotation_file.endswith('.json'):
            # data has two components:
            #  'questions': 19242 items,
            #       each cotaining question_type_id, question_id, question, data_type, data_id, choice_a, choice_b, choice_c, choice_d, answer
            #  'question_type':  question_type to question_type_id dict
            data = json.load(open(args.annotation_file))
            ques_type_id_to_name = {id: n for n, id in data['question_type'].items()}



    results = eval_single(args.result_file, eval_dataset=args.eval_dataset, annotation_file=args.annotation_file )
    if args.eval_dataset == 'SEED' and not 'seedbench2' in args.annotation_file:
        eval_single(args.result_file, eval_dataset=args.eval_dataset, eval_only_type='image', annotation_file=args.annotation_file)
        eval_single(args.result_file, eval_dataset=args.eval_dataset, eval_only_type='video', annotation_file=args.annotation_file)

        if False:
            with open(args.result_upload_file, 'w') as fp:
                for question in data['questions']:
                    qid = question['question_id']
                    if qid in results:
                        result = results[qid]
                    else:
                        result = results[int(qid)]
                    fp.write(json.dumps({
                        'question_id': qid,
                        'prediction': result['text']
                    }) + '\n')
