# InstructionERC


## 🎥 Overview
This is the repository for our work **SKP**, which is recieved by **CIKM 2023**.

## Introduction
We propose a **S**tructured **K**nowledge-aware **P**re-training method (**SKP**). In the pre-training stage, we introduce two novel structured knowledge-aware tasks, guiding the model to effectively learn the implicit relationship and better representations of complex subgraphs. 
In downstream KBQA task, we further design an efficient linearization strategy and an interval attention mechanism, which assist the model to better encode complex subgraphs and shield the interference of irrelevant subgraphs during reasoning respectively.
Detailed experiments and analyses on WebQSP verify the effectiveness of SKP, especially the significant improvement in subgraph retrieval (+4.08% H@10).

## 🍯 Overall Framework
<img width="733" alt="image" src="https://github.com/dongguanting/Structured-Knowledge-aware-Pretraining-for-KBQA/assets/60767110/c63e55fb-0cee-474c-8dbf-392498ad24e6">

## 🎯 Quick Start

Our work is built on the [UniK-QA](https://github.com/facebookresearch/UniK-QA) framework.


### Dependencies

General Setup Environment:
- Python 3
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (currently tested on version 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (version 3.0.2, unlikely to work with a different version)

DPR Setup Environment:
```
cd ./KBQA/DPR-main/
pip3 install -r requirements.txt
```

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



## 📋 Result:

### Main Result

| Model                                                    |  Hits@1    | 
| --------------------- | :------:   | 
| GraftNet                                                 |  69.5     | 
| PullNet                                                    |  68.1    | 
| EMQL                                                      |  75.5     | 
| BERT-KBQA                                                |  72.9    | 
| NSM                                                       |  74.3    | 
| KGT5                                                      | 56.1 | 
| SR-NSM                                                   | 69.5| 
| EmbededKGQA                                               | 72.5| 
| DECAF(Answer only)                                        | 74.7 | 
| UniK-QA∗                                                 | 77.9 |
| SKP (ours)                                               | **79.6** | 


### In-Context Learning Result For LLMs

Since there were very few open source large models when the article was written (2022.12), we now supplement the SKP framework with the results of **In Context Learning** when the LLMs is used as a Reader. Due to the limitation of the Max sequence length of the LLMs, for the **Topk** documents retrieved by the retriever, we select the documents with the highest semantic similarity and truncate them with 2048 tokens as the knowledge prompting for reader (about 5 documents)


| Model                                                    |  Hits@1     | 
| -------------------------------------------------------- | :------:   | 
| SKP(ChatGPT)                                              |  69.2     | 
| SKP(LLAMA)                                                |  16.8   | 
| SKP(LLAMA2)                                                |  coming soon     | 
| SKP(ChatGLM)                                                |  coming soon     | 
| SKP(ChatGLM2)                                               |  coming soon     | 


### Supervised Finetuning Result For LLMs 

SFT

| Model                                                    |  Hits@1     | 
| -------------------------------------------------------- | :------:   | 
| SKP(LLAMA)                                                |  coming soon   | 
| SKP(LLAMA2)                                                |  coming soon     | 
| SKP(ChatGLM)                                                |  coming soon     | 
| SKP(ChatGLM2)                                               |  coming soon     | 



