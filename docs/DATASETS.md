
# Data Preparation

### Visual Instruction Tuning Data

#### LLaVA 1.5 Visual Instruction Tuning Data
- Annotation:
  - [LLaVA 1.5 mix 665K data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) in json file
- Image sources:
  - COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
  - GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
  - OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
  - TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
  - VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

Pull all image folders into a single directory `llava_imgs`, i.e.
```js
llava_imgs/
|–– coco/
|–– gqa/
|–– ocr_vqa/
|–– textvqa/
|–– vg/
```

#### CaD Visual Instruction Tuning Phase-1 Data
- Annotation
    - [CaD-VI Phase-1 data](https://huggingface.co/datasets/wlin21at/CaD-Inst/blob/main/json_files/phase1_instruct_data_278k.json) in json file
- Image sources
  - [Open Images v6](https://storage.googleapis.com/openimages/web/download_v6.html)
  - [COCO 2017](https://cocodataset.org/#download)
  - [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
  - [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)

Put all image files in a single directory `localized_narratives_imgs`, i.e.

```js
localized_narratives_imgs/
|–– 7df2bc41b2093a4e.jpg
|–– 51fc0a698e989db2.jpg
|–– 000000292416.jpg
|–– 82a41716439ee9e1.jpg
|–– 300839715.jpg
|–– ADE_train_00009414.jpg
|–– 27323284.jpg
|–– 4950626600.jpg
|–– ...
```

We will also prepare to host the 278K image pairs (556K images in total) used in Phase-1 soon. 


#### CaD Visual Instruction Tuning Phase-2 Data
- Annotation:
  - [CaD-VI Phase-2 data](https://huggingface.co/datasets/wlin21at/CaD-Inst/blob/main/json_files/phase2_instruct_data_71k.json) in json file
- Image sources:
  - COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip) (downloaded before)

### Evaluation Data

#### BISON
- Annotation:
  - [BISON annot data](../eval_json_files/BISON.jsonl) in jsonl file
- Image sources:
  - [BISON images](https://mail2sysueducn-my.sharepoint.com/personal/huangyp28_mail2_sysu_edu_cn/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhuangyp28%5Fmail2%5Fsysu%5Fedu%5Fcn%2FDocuments%2Frelease%2FSparkles%2Fevaluation%2FBISON%2Fimages), provided by [SparklesChat](https://github.com/HYPJUDY/Sparkles?tab=readme-ov-file)
#### SVO Probes
- Annotation:
  - [SVO Probes annot data](../eval_json_files/SVO_Probes.jsonl) in jsonl file
- Image sources:
  - [SVO Probes images](https://github.com/google-deepmind/svo_probes/blob/main/image_urls.txt)
#### NLVR2
- Annotation:
  - [NLVR2 annot data](../eval_json_files/NLVR2.jsonl) in jsonl file
- Image sources:
  - [NLVR2 images](https://mail2sysueducn-my.sharepoint.com/personal/huangyp28_mail2_sysu_edu_cn/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhuangyp28%5Fmail2%5Fsysu%5Fedu%5Fcn%2FDocuments%2Frelease%2FSparkles%2Fevaluation%2FNLVR2%2Fimages), provided by [SparklesChat](https://github.com/HYPJUDY/Sparkles?tab=readme-ov-file)
#### EQBEN
- Annotation:
  - [EQBEN annot data](../eval_json_files/EQBEN.jsonl) in jsonl file
- Image sources:
  - [EQBEN images](https://entuedu-my.sharepoint.com/personal/tan317_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Ftan317%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FPh%2ED%2D2%2Fintern%2Fmicrosoft%2Fwork%2Fgithub%2Feval%5Fminigpt4%2Fdata%2Feqben%5Fvllm%2Ezip&parent=%2Fpersonal%2Ftan317%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FPh%2ED%2D2%2Fintern%2Fmicrosoft%2Fwork%2Fgithub%2Feval%5Fminigpt4%2Fdata&ga=1)
#### COLA
- Annotation:
  - [COLA annot data](../eval_json_files/COLA.jsonl) in jsonl file
- Image sources:
  - [COLA images](https://github.com/arijitray1993/COLA/blob/main/data/COLA_multiobjects_matching_benchmark.json)
#### CaD-QA
- Annotation:
  - [CaD-QA annot data](https://huggingface.co/datasets/wlin21at/CaD-Inst/blob/main/json_files/CaD_QA_eval.jsonl) in jsonl file
- Image sources:
  - VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) (downloaded before)