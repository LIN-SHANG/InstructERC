# 🔥 InstructERC

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instructerc-reforming-emotion-recognition-in/emotion-recognition-in-conversation-on-4)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on-4?p=instructerc-reforming-emotion-recognition-in)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instructerc-reforming-emotion-recognition-in/emotion-recognition-in-conversation-on)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on?p=instructerc-reforming-emotion-recognition-in)	

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/instructerc-reforming-emotion-recognition-in/emotion-recognition-in-conversation-on-meld)](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on-meld?p=instructerc-reforming-emotion-recognition-in)

## 🎥 Overview

This repository contains the open-sourced official implementation of our work **InstructERC**:

[InstructERC: Reforming Emotion Recognition in Conversation with Multi-task Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2309.11911)


If you find this repo helpful, please cite the following paper:

```bibtex
@article{lei2023instructerc,
  author       = {Shanglin Lei and
                  Guanting Dong and
                  Xiaoping Wang and
                  Keheng Wang and
                  Sirui Wang},
  title        = {InstructERC: Reforming Emotion Recognition in Conversation with a
                  Retrieval Multi-task LLMs Framework},
  journal      = {CoRR},
  volume       = {abs/2309.11911},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2309.11911},
  doi          = {10.48550/ARXIV.2309.11911},
  eprinttype    = {arXiv},
  eprint       = {2309.11911},
  timestamp    = {Tue, 30 Jan 2024 15:46:48 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2309-11911.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}


```

## Introduction
In this study, we propose a novel approach, namely **InstructERC, to reformulates the ERC task from a discriminative framework to a generative framework based on LLMs.** InstructERC has two significant contributions: Firstly, InstructERC introduces a simple yet effective retrieval template module, which helps the model explicitly integrate multi-granularity dialogue supervision information by concatenating the historical dialog content, label statement, and emotional domain demonstrations with high semantic similarity. Furthermore, we introduce two additional emotion alignment tasks, namely speaker identification and emotion prediction tasks, to implicitly model the dialogue role relationships and future emotional tendencies in conversations. **Our LLM-based plug-and-play plugin framework significantly outperforms all previous models and achieves comprehensive SOTA on three commonly used ERC datasets.** Extensive analysis of parameter-efficient and data-scaling experiments provide empirical guidance for applying InstructERC in practical scenarios. Our code will be released after blind review.

## 🍯 Overall Framework
![image](https://github.com/user-attachments/assets/c6fb1704-223d-4e6f-9d0c-492a3bdf720e)


## 🎯 Quick Start


This repo consists of following files:
```plain
.
├── checkpoint
├── code
│   ├── data_process_mixed.py
│   ├── data_process_plain.py
│   ├── data_process.py
│   ├── data_utils
│   ├── main_new.py
│   ├── train_and_inference_Mixed.sh
│   ├── train_and_inference_Plain.sh
│   └── train_and_inference_Uni.sh
├── data
│   ├── EmoryNLP
│   ├── iemocap
│   └── meld
├── demo
│   └── demo.ipynb
├── envs
│   └── requirements.txt
├── experiments
├── file_structure.txt
├── LLM_bases
│   ├── Bloom-560m
│   ├── ChatGLM
│   ├── ChatGLM2
│   ├── LLaMA
│   └── LLaMA2
├── original_data
│   ├── dailydialog
│   ├── EmoryNLP
│   ├── iemocap
│   ├── meld
│   └── peek_of_dataset.ipynb
└── README.md
```
<!-- 正如以上树状结构的形式展示目录和文件的层级关系所示,InstructERC由code,data,demo, envs,LLM_bases和original_data文件夹组成, 
- code 存放了InstructERC所有可以执行的代码,包括不同方式的数据处理data_process.py (mixed, plain),主程序文件main_new.py,data_utils和控制整个流程的bash文件train_and_inference_Uni.sh (Plain, Mixed).

- data存放由original data和data_process.py脚本处理好的,直接输入给LLM的数据.

- demo存放了能够直接加载并运行我们在特定数据上已完成finetune的LLM进行推理的脚本demo.ipynb,由jupyter notebook打开并运行.

- envs存放了本项目所需要的相关依赖,环境和库,我们强烈建议您使用docker并新建一个conda的虚拟环境,以避免对您之前的环境和文件产生影响.

- LLM_bases存放了官方提供的LLMs原件,我们可以在huggingface上面下载这些原件.

- orignal_data存放的是由COSMIC团队处理好的,被广泛使用的数据集,包含IEMOCAP,MELD,EmoryNLP和dailydialog. -->

As shown in the tree-like structure above, InstructERC consists of the following folders: code, data, demo, envs, LLM_bases, and original_data.
- The checkpoint folder is created to storage InstructERC'checkpoint.
(Pertraining Checkpoint, Supervised Finetuning Checkpoint at each epoch)

- The code folder contains all the executable code for InstructERC, including data processing scripts such as data_process.py (mixed, plain), the main program file main_new.py, data_utils, and the bash script train_and_inference_Uni.sh (Plain, Mixed) that controls the entire workflow.

- The data folder stores the data that has been processed by the data_process.py script from the original data. This processed data can be directly fed into the LLM (Large Language Model) as input.

- The demo folder contains the script demo.ipynb, which can be loaded and run directly on the LLM that has been finetuned on specific data. It can be opened and run using Jupyter Notebook.

- The envs folder contains the relevant dependencies, environments, and libraries required for this project. We strongly recommend using Docker and creating a new Conda virtual environment to avoid affecting your existing environment and files.

- The experiments folder is created to storage the result under different experiment's settings.

- The LLM_bases folder stores the original models provided by the official LLMs. These models can be downloaded from Hugging Face.

- The original_data folder contains the widely used datasets processed by the COSMIC team, including IEMOCAP, MELD, EmoryNLP, and dailydialog. We write a script for you to have a peek of these datasets, namely peek_of_dataset.ipynb.

### Dependencies
We suggest you create a docker environment for InstructERC to ensure that your previous systems, libraries and files are not effected.
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

### LLMs download
Follow this [link](https://huggingface.co/docs/hub/models-downloading).

### ONLY Validate Our Work
Due to Meituan's code review process, the public release date of the model parameters is unpredictable. However, based on our tests on other machines, we have achieved performance results that fluctuate by ±0.5 compared to the data presented in the paper. Additionally, for the implementation of the demonstration, you can refer to the following link:
https://github.com/UKPLab/sentence-transformers

### Completely repeat all our work
```
cd ./InstructERC
```

To reproduce the results, we have three pipelines available.

- If you want to reproduce the Main Result Reproduction, you can run the ***train_and_inference_Uni.sh***

```
bash train_and_inference_Uni.sh
```



The Shellparameter that controls the mainprocess: Flag
```
the value of which is 0 or 1. 
The mainprocess will interrupt when flag is 0
```


The hyperparameters you need setting: 
```
1.MODEL_NAME (selections: ChatGLM, ChatGLM2, LLaMA, LLaMA2)
# MODEL_NAME determines on which model base InstructERC will be fine-tuned.

2.Experiments_setting (selections: LoRA, All-parameters)
# The Experiments_setting parameter determines whether it is full parameter fine-tuning or efficient parameter fine-tuning.

3.dataset (selections: IEMOCAP, MELD, EmoryNLP)
# The specific dataset you want InstructERC to finetune on.

4.accumulations (type:int)
# Due to the limitations of the GPU, we have chosen the method of gradient accumulation for fine-tuning.

5.graphics_card (type:int)
# The graphics_card represents the number of graphics cards you use when fine-tuning.

Notes: batch size = graphics_card * accumulations
```

The remaining subprocesses determined by these hyperparameters are designed to conduct different experiments.

---

- If you want to reproduce the Unified dataset Experiment, you can run the ***train_and_inference_Mixed.sh***

```
bash train_and_inference_Mixed.sh
```
Compared to train_and_inference_Uni.sh, you should overlook the hyperparameter dataset due to the unified dataset including all ERC datasets.

---
- If you want to reproduce the LoRA+Backbone Experiment, you can run the ***train_and_inference_Plain.sh***

```
bash train_and_inference_Mixed.sh
```



## 📋 Result:

### Main Result
<center>
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
| Llama2 | **71.39**| **69.15**| **41.37**| **60.64**| LLM|
</center>

### All Parameters vs Parameter Efficiency
In order to investigate the effect of different parameter fine-tuning methods on the ERC task, we conducted comparative experiments in Table 2.

<center>
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
</center>


### A.1 Unified dataset labeling

We continue to use the previous datasets IEMOCAP, MELD, and EmoryNLP.
In accordance with The Feeling Wheel [^1] proposed in 1982, as shown in Figure 2, we align all emotional labels of three datasets under this standard, the details of which are shown in Table 3.
After completing the label mapping, there are a total of 9 kinds of emotional labels, which are joyful, sad, neutral, mad, excited, powerful, fear, peaceful, and disgust.


Figure2: The Feeling Wheel[^1]
<img width="1188" alt="image" src="https://github.com/LIN-SHANG/InstructERC/assets/48848639/a44d9501-49f0-4d0e-bbf2-d2e26d16bc4f">




<center>
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
</center>


### A.2 Unified dataset Experiment
We still utilize the LoRA method in PEFT to train InstructERC on the unified dataset, and the training results are evaluated on the three datasets respectively. Meanwhile, we design total mix and ratio mix experiments to explore the impact of different data mixing strategies and data quantities on the model. On below basis, we further explore the impact of data sampling ratio on the model's performance.
The details are shown in the Table 5, a more intutive presentation is shown in Figure 6.

<center>

| Data Precent | IEMOCAP W-F1 (Total Mix) | IEMOCAP W-F1 (Ratio Mix) | IEMOCAP W-F1 (Single) | MELD W-F1 (Total Mix) | MELD W-F1 (Ratio Mix) | MELD W-F1 (Single) | EmoryNLP W-F1 (Total Mix) | EmoryNLP W-F1 (Ratio Mix) | EmoryNLP W-F1 (Single) |
| :------------: | ----------------------- | ----------------------- | --------------------- | --------------------- | --------------------- | ------------------ | ------------------------ | ------------------------ | ---------------------- |
| 1            | 68.99                   | 68.99                   | **71.39**             | 68.07                 | 68.07                 | **69.15**          | 40.27                    | 40.27                    | **41.37**              |
| 1/2          | 67.95                   | 68.96                   | **69.13**             | 66.50                 | 66.42                 | **67.54**          | 39.18                    | 39.33                    | **39.65**              |
| 1/4          | 63.02                   | 64.46                   | **67.54**             | 66.41                 | 65.85                 | **66.42**          | 38.26                    | 37.29                    | **38.33**              |
| 1/8          | 58.48                   | 60.06                   | **64.13**             | 64.57                 | 62.94                 | **65.14**          | 38.27                    | **39.24**                | 38.24                  |
| 1/16         | 57.77                   | 53.40                   | **60.42**             | 61.15                 | 58.42                 | **62.89**          | 37.19                    | **37.60**                | 36.83                  |
| 1/32         | 45.89                   | 48.50                   | **54.76**             | 57.38                 | **57.76**             | 57.72              | **37.09**                | 36.09                    | 34.03                  |
| 1/64         | 38.42                   | **43.07**               | 30.34                 | **54.26**             | 53.29                 | 45.48              | **35.19**                | 34.65                    | 26.10                  |
</center>



[^1]: Willcox, K. (1982). The Feeling Wheel. Journal of Counseling & Development, 61(3), 191-193.

