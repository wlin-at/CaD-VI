#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for CKPT in your_model_name
do
  model_path=/path/to/$CKPT
  if [[ $CKPT == *"13b"* ]];
  then
    model_base=/path/to/vicuna-13b-v1.5/
  else
    model_base=/path/to/vicuna-7b-v1.5/
  fi
  question_file_path=./eval_json_files/SVO_Probes.jsonl
  image_folder_dir=/path/to/SVO_Probes_images/
  conv_mode=vicuna_v1
  inference_mode=generate
  eval_dataset=common_diff
  result_main_dir=/path/to/SVO_Probes_results
  mkdir -p $result_main_dir

  answers_dir=$result_main_dir
  results_updaload_dir=$result_main_dir/answers_upload
  for IDX in $(seq 0 $((CHUNKS-1))); do
      CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
          --model-path $model_path \
          --model-base $model_base \
          --question-file $question_file_path \
          --image-folder $image_folder_dir \
          --answers-file ${answers_dir}/$CKPT/${CHUNKS}_${IDX}.jsonl \
          --num-chunks $CHUNKS \
          --chunk-idx $IDX \
          --temperature 0 \
          --conv-mode $conv_mode \
          --inference_mode $inference_mode \
          --eval_dataset $eval_dataset &

  done
  wait
  output_file=${answers_dir}/$CKPT/merge.jsonl

  # Clear out the output file if it exists.
  > "$output_file"

    # Loop through the indices and concatenate each file.
  for IDX in $(seq 0 $((CHUNKS-1))); do
      cat ${answers_dir}/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
  done

  python utils/compute_eval_acc/compute_binaryimgselect_acc.py --annot_file $question_file_path --result_file $output_file --question_category true
done
