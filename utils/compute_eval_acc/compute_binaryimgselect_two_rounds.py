





import json
import os.path as osp

from argparse import ArgumentParser
def eval_single(data, results, question_category=True):
    # n_samples = len(results) // 2
    # total_correct = 0
    # for idx in range(n_samples):
    #     row0 = results[f'{idx}_0']
    #     pred_answer0 = row0['text']
    #     gt_answer0 = data[f'{idx}_0']['answer']
    #
    #     row1 = results[f'{idx}_1']
    #     pred_answer1 = row1['text']
    #     gt_answer1 = data[f'{idx}_1']['answer']
    #
    #     if pred_answer0 == gt_answer0 and pred_answer1 == gt_answer1:
    #         total_correct += 1
    # total_accuracy = total_correct / float(n_samples) * 100
    # print(f"Total_accuracy: {total_accuracy:.2f}%", file=f_write)


    correct_counts = {}
    total_counts = {}
    n_samples = len(results) // 2
    for idx in range(n_samples):
        if question_category:
            question_type = results[f'{idx}_0']['image'][0].split('/')[0]
        else:
            question_type = 'all'
        total_counts[question_type] = total_counts.get(question_type, 0) + 1

        row0 = results[f'{idx}_0']
        pred_answer0 = row0['text']
        gt_answer0 = data[f'{idx}_0']['answer']

        row1 = results[f'{idx}_1']
        pred_answer1 = row1['text']
        gt_answer1 = data[f'{idx}_1']['answer']
        if pred_answer0 == gt_answer0 and pred_answer1 == gt_answer1:
            correct_counts[question_type] = correct_counts.get(question_type, 0) + 1
    for key in total_counts.keys():
        if key not in correct_counts:
            correct_counts[key] = 0
    total_count = 0
    total_correct = 0
    for question_type in sorted(correct_counts.keys()):
        accuracy = correct_counts[question_type] / total_counts[question_type] * 100
        print(f"{question_type}: {accuracy:.2f}%", file=f_write)
        total_count += total_counts[question_type]
        total_correct += correct_counts[question_type]
    total_accuracy = total_correct / total_count * 100
    print(f"Total_accuracy: {total_accuracy:.2f}%", file=f_write)
    # for question_id, row in results.items():
    #     if question_category:
    #         question_type = results[question_id]['image'][0].split('/')[0]
    #     else:
    #         question_type = 'all'
    #
    #     total_counts[question_type] = total_counts.get(question_type, 0) + 1
    #     pred_answer = row['text']
    #     gt_answer = data[question_id]['answer']
    #     if pred_answer == gt_answer:
    #         correct_counts[question_type] = correct_counts.get(question_type, 0) + 1
    # total_count = 0
    # total_correct = 0
    # for question_type in sorted(correct_counts.keys()):
    #     accuracy = correct_counts[question_type] / total_counts[question_type] * 100
    #     print(f"{question_type}: {accuracy:.2f}%", file=f_write)
    #     total_count += total_counts[question_type]
    #     total_correct += correct_counts[question_type]
    # total_accuracy = total_correct / total_count * 100
    # print(f"Total_accuracy: {total_accuracy:.2f}%", file=f_write)



if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("--annot_file", type=str, default='')
    args.add_argument("--result_file", type=str, default='')
    args.add_argument("--question_category", type=bool, default=False)
    args = args.parse_args()
    annot_file = args.annot_file
    result_file = args.result_file
    question_category = args.question_category

    # for result_file in result_file_list:
    print_file = osp.join( '/'.join(result_file.split('/')[:-1]), 'text2image_retrieval_acc.txt'   )
    f_write = open(print_file, 'w')
    # data = json.load(open(annot_file))
    data = {}
    for line_id, line in enumerate(open(annot_file)):
        row = json.loads(line)
        data[row['question_id']] = row
    results = {}
    for line_id, line in enumerate(open(result_file)):
        row = json.loads(line)
        results[row['question_id']] = row
    eval_single(data, results, question_category=question_category)
    f_write.close()


