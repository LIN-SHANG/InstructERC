# InstructionERC

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instructerc-reforming-emotion-recognition-in/emotion-recognition-in-conversation-on-4)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on-4?p=instructerc-reforming-emotion-recognition-in)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instructerc-reforming-emotion-recognition-in/emotion-recognition-in-conversation-on)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on?p=instructerc-reforming-emotion-recognition-in)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instructerc-reforming-emotion-recognition-in/emotion-recognition-in-conversation-on-meld)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on-meld?p=instructerc-reforming-emotion-recognition-in)

## 🎥 Overview

This repository contains the open-sourced official implementation of our work **InstructERC**:

[InstructERC: Reforming Emotion Recognition in Conversation with a Retrieval Multi-task LLMs Framework](https://arxiv.org/abs/2309.11911)


If you find this repo helpful, please cite the following paper:

```bibtex
@misc{lei2023instructerc,
      title={InstructERC: Reforming Emotion Recognition in Conversation with a Retrieval Multi-task LLMs Framework}, 
      author={Shanglin Lei and Guanting Dong and Xiaoping Wang and Keheng Wang and Sirui Wang},
      year={2023},
      eprint={2309.11911},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Introduction
In this study, we propose a novel approach, namely **InstructERC, to reformulates the ERC task from a discriminative framework to a generative framework based on LLMs.** InstructERC has two significant contributions: Firstly, InstructERC introduces a simple yet effective retrieval template module, which helps the model explicitly integrate multi-granularity dialogue supervision information by concatenating the historical dialog content, label statement, and emotional domain demonstrations with high semantic similarity. Furthermore, we introduce two additional emotion alignment tasks, namely speaker identification and emotion prediction tasks, to implicitly model the dialogue role relationships and future emotional tendencies in conversations. **Our LLM-based plug-and-play plugin framework significantly outperforms all previous models and achieves comprehensive SOTA on three commonly used ERC datasets.** Extensive analysis of parameter-efficient and data-scaling experiments provide empirical guidance for applying InstructERC in practical scenarios. Our code will be released after blind review.

## 🍯 Overall Framework
![image](https://github.com/LIN-SHANG/InstructERC/assets/60767110/c0cc9d87-2bea-4783-97be-f1d319c61ec3)


## 🎯 Quick Start

<!-- Our work is built on the [UniK-QA](https://github.com/facebookresearch/UniK-QA) framework. -->
```plain


```


此Repo由以下文件组成：
- code
- demo
- data
- data_utils
- envs
- experiments
- LLM_bases
- Original_data
- README.md

### Dependencies
We suggest you create a docker envirment for InstructERC to ensure that your previous systems, libraries and files are not effected.
Make sure your Devtoolset-8-toolchain' version align with us:
```
yum install devtoolset-8-toolchain
yum install gcc 9.3.1
```

General Setup Environment:
- Python 3
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (currently tested on version 2.0.0)
- [Transformers](http://huggingface.co/transformers/) (version 4.30.2, unlikely to work with a different version)

InstructERC Setup Environment:
```
cd ./InstructERC/envs/
pip3 install -r requirements.txt
```


<!-- 
### Data Preprocessing

```
wget https://dl.fbaipublicfiles.com/UniK-QA/data.tar.xz
tar -xvf data.tar.xz
```

Prepare the above data, and we provide two linearization methods:
- Normal linearization：

```
cd ./data_process/
python webqsp_preprocess.py
```

- Linearization of merging complex subgraphs：

```
cd ./data_process/
python webqsp_preprocess_complex2.py
```

Our code is thoroughly commented! The final output will consist of three TSV files for encoding.


### Pretraining DPR：

We use linearized subgraphs to perform structure knowledg aware pretraining on the processed TSV files.

1. First, we randomly extract 1 million subgraphs from the preprocessed TSV files:

```
bash random_sample_complex1.sh
```

- Our 1 million subgraphs can be directly downloaded [here](https://drive.google.com/drive/folders/1UnWOB0zApioYOJ4GuS3JKSWAkkeDXxQv?usp=drive_link).

2. For DPR pretraining, we provide 3 modes:

- Joint pretraining for **Mask Language Modeling** and **Contrastive Learning**

```
cd ./DPR_pretraining/bash/
bash train_mlm_contrastive_mask.sh
```

- Only Mask Language Modeling:

```
bash train-mlm.sh
```

- Only Contrastive Learning

```
bash train-contrastive.sh
```



### Training DPR：
Due to the pretraining process, we first load the checkpoint for structured pretraining, and then train DPR:

```
cd ./DPR-main/
bash train_encoder1.sh
```

The detailed information can be referred to the GitHub repository of DPR. [DPR](https://github.com/facebookresearch/DPR)


### Encoding TSV into embedding files:

Using the trained DPR, encode the three TSV files into embedding vector files. The file "all_relations.tsv" is split into 100 parts for encoding, and this process takes a long time.

```
cd ./DPR-main/
for id in {1..10..1} 
   bash gen_all_relation_emb${id}.sh
bash gen_condense_hyper_relation_emb.sh
bash gen_condense_hyper_relation_emb.sh
```

In each bash command:
- WEBQSP_DIR is your base path.
- model_dir is the path to your DPR checkpoint.
- out_dir is the path to the output directory for the encoded embeddings.



### Preprocessing the input data for FID:

Using FAISS, filter out the top-k subgraphs corresponding to each question from the generated subgraph embeddings in the previous step.

```
python dpr_inference.py
```

After generating the DPR output data, further filtering and conversion into the format compatible with FID can be done using fid_preprocess.py.

```
python fid_preprocess.py
```

Our Subgraph Retrieval results are shown here：
![image](https://github.com/dongguanting/SKP-for-KBQA/assets/60767110/fad13037-4b5c-46f5-abd6-424f1bc0d731)



### Training and Testing with FiD:

Next, the input to the [FiD](https://github.com/facebookresearch/FiD) reader is created for each question using the most relevant relations retrieved by DPR.Finally, a FiD model can be trained using the SKP input. 

If you want to reproduce the results for inference directly, our FID inputs and model have been made publicly available. 
- Our FID input can be downloaded [here](https://drive.google.com/drive/folders/1UnWOB0zApioYOJ4GuS3JKSWAkkeDXxQv?usp=drive_link).
- Our trained FiD checkpoint can be downloaded [here](https://drive.google.com/drive/folders/1UnWOB0zApioYOJ4GuS3JKSWAkkeDXxQv?usp=drive_link). (Our model was trained in late 2020, so you may need to check out an older version of FiD.)


Train FiD

```
python -u train.py \
  --train_data_path {data dir}/webqsp_train.json \
  --dev_data_path {data dir}/webqsp_dev.json \
  --model_size large \
  --per_gpu_batch_size 1 \
  --n_context 100 \
  --max_passage_length 200 \
  --total_step 100000 \
  --name {checkpoint name} \
  --model_path {loading backbone model path} \
  --checkpoint_dir {save path} \
  --eval_freq 250 \
  --eval_print_freq 250
```

Inference FiD

```
python test.py \
  --model_path {checkpoint path} \
  --test_data_path {data path}/webqsp_test.json \
  --model_size large \
  --per_gpu_batch_size 4 \
  --n_context 100 \
  --name {checkpoint name} \
  --checkpoint_dir {base dir}/FiD-snapshot_nov_2020 \
```

Our Final Result：

```
2022-12-26 11:43:51 | WARNING | __main__ | 0, total 1639 -- average = 0.796
2022-12-26 11:43:51 | INFO | __main__ | total number of example 1639
2022-12-26 11:43:51 | INFO | __main__ | EM 0.795812
```


-->
## 📋 Result:

### Main Result
Table1: The main result on three benchmarks

| Dataset          | IEMOCAP | MELD  | EmoryNLP | Average | type|
|:----------------:|---------|-------|----------|---------|:-----:|
| Models           | W-F1    | W-F1  | W-F1     | W-F1    |
| **Discriminant Models**        |
| EmotionIC  | 69.50   | 66.40 | **40.01**| **58.63**|Attention|
| SACL        | 69.22   | **66.45**| 39.65 | 58.44   |Recurrent|
| SKAIG       | 66.98   | 65.18 | 38.88    | 57.01   |Knowledge|
| GraphCFC    | 68.91   | 58.86 | -        | -       |Graph|
| UniMSE      | **70.66**| 65.51 | -        | -       |Multimodel|
| **Zero-shot + InstructERC**         |
| ChatGLM          | **38.6**| **38.8**| 19.6   | **32.33**| LLM|
| ChatGLM2         | 21.1    | 21.8  | **24.4**| 22.43   | LLM|
| Llama            | 0.753   | 9.12  | 5.31    | 5.06    | LLM|
| Llama2           | 2.774   | 16.28 | 8.36    | 9.46    | LLM|
| **LoRA + InstructERC**   |
| ChatGLM| 36.04   | 46.41 | 30.86    | 37.77   | LLM|
| ChatGLM2| 67.54  | 65.58 | 39.09    | 57.40   | LLM|
| Llama  | 64.17   | 67.62 | 39.34    | 57.04   | LLM|
| Llama2 | **71.39**| **69.15**| **41.37**| **60.64**|

### All Parameters vs Parameter Efficiency
In order to investigate the effect of different parameter fine-tuning methods on the ERC task, we conducted comparative experiments in Table 2.

Table 2: The comparison results of different parameter fine-tuning settings on three benchmarks.
| Dataset     | IEMOCAP | MELD   | EmoryNLP | Average |
|:-----------:|---------|--------|----------|---------|
| Models      | W-F1    | W-F1   | W-F1     | W-F1    |
| **All parameters + InstructERC** |       |        |          |         |
| ChatGLM  | 33.94   | 37.96  | 13.25    | 28.38   |
| ChatGLM2 | 70.05   | 63.24  | 38.77    | 57.35   |
| Llama    | 69.38   | **66.01**  | **40.21**    | **58.53**   |
| Llama2   | **70.30**   | 64.80  | 40.05    | 58.38   |
| **LoRA + InstructERC** |       |        |          |         |
| ChatGLM  | 36.04   | 46.41  | 30.86    | 37.77   |
| ChatGLM2 | 67.54   | 65.58  | 39.09    | 57.40   |
| Llama    | 69.71   | 68.89  | 39.90    | 59.50   |
| Llama2   | **71.39**   | **69.15**  | **41.37**    | **60.64**   |



### A.1 Unified dataset labeling

We continue to use the previous datasets IEMOCAP, MELD, and EmoryNLP.
In accordance with The Feeling Wheel [^1] proposed in 1982, as shown in Figure 2, we align all emotional labels of three datasets under this standard, the details of which are shown in Table 3.
After completing the label mapping, there are a total of 9 kinds of emotional labels, which are joyful, sad, neutral, mad, excited, powerful, fear, peaceful, and disgust.


Figure2: The Feeling Wheel[^1]
<img width="668" alt="image" src="https://github.com/LIN-SHANG/InstructERC/assets/48848639/65526715-d02f-41f6-b0d8-3e5da7311e3f">


Table 3: Unified Label Mapping

| Number | IEMOCAP | MELD | EmoryNLP | Final Emotion |
| :------: | :-------: | :----: | :--------: | :-------------: |
| 1      | happy   | joyful | joyful   | joyful        |
| 2      | sad     | sad    | sad      | sad           |
| 3      | neutral | neutral | neutral | neutral       |
| 4      | angry   | angry  | mad      | mad           |
| 5      | excited | N/A    | N/A      | excited       |
| 6      | N/A     | surprise | powerful | powerful      |
| 7      | scared  | fear   | frustrated | fear          |
| 8      | N/A     | N/A    | peaceful | peaceful      |
| 9      | N/A     | disgust | N/A      | disgust       |


### A.2 Unified dataset Experiment
We still utilize the LoRA method in PEFT to train InstructERC on the unified dataset, and the training results are evaluated on the three datasets respectively. Meanwhile, we design total mix and ratio mix experiments to explore the impact of different data mixing strategies and data quantities on the model. On below basis, we further explore the impact of data sampling ratio on the model's performance.
The details are shown in the Table 4, a more intutive presentation is shown in Figure 3.

Table 4: The Unified Dataset Experiments of Llama2 on three benchmarks
| Data Precent | IEMOCAP W-F1 (Total Mix) | IEMOCAP W-F1 (Ratio Mix) | IEMOCAP W-F1 (Single) | MELD W-F1 (Total Mix) | MELD W-F1 (Ratio Mix) | MELD W-F1 (Single) | EmoryNLP W-F1 (Total Mix) | EmoryNLP W-F1 (Ratio Mix) | EmoryNLP W-F1 (Single) |
| :------------: | ----------------------- | ----------------------- | --------------------- | --------------------- | --------------------- | ------------------ | ------------------------ | ------------------------ | ---------------------- |
| 1            | 68.99                   | 68.99                   | **71.39**             | 68.07                 | 68.07                 | **69.15**          | 40.27                    | 40.27                    | **41.37**              |
| 1/2          | 67.95                   | 68.96                   | **69.13**             | 66.50                 | 66.42                 | **67.54**          | 39.18                    | 39.33                    | **39.65**              |
| 1/4          | 63.02                   | 64.46                   | **67.54**             | 66.41                 | 65.85                 | **66.42**          | 38.26                    | 37.29                    | **38.33**              |
| 1/8          | 58.48                   | 60.06                   | **64.13**             | 64.57                 | 62.94                 | **65.14**          | 38.27                    | **39.24**                | 38.24                  |
| 1/16         | 57.77                   | 53.40                   | **60.42**             | 61.15                 | 58.42                 | **62.89**          | 37.19                    | **37.60**                | 36.83                  |
| 1/32         | 45.89                   | 48.50                   | **54.76**             | 57.38                 | **57.76**             | 57.72              | **37.09**                | 36.09                    | 34.03                  |
| 1/64         | 38.42                   | **43.07**               | 30.34                 | **54.26**             | 53.29                 | 45.48              | **35.19**                | 34.65                    | 26.10                  |

Figure 3: The low resourece exploring on three benchmarks using different data mixing strategies
<img width="984" alt="image" src="https://github.com/LIN-SHANG/InstructERC/assets/48848639/a9a734b7-a893-4ec1-b067-dbebe6cd2ccd">



[^1]: Willcox, K. (1982). The Feeling Wheel. Journal of Counseling & Development, 61(3), 191-193.

