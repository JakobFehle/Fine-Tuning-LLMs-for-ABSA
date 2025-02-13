# Leveraging Fine-Tuning of Large Language Models for Aspect-Based Sentiment Analysis in Resource-Scarce Environments (Work in Progress)

## Overview

This repository contains the code and evaluation scripts for our study on fine-tuning open-source Large Language Models (LLMs) for Aspect-Based Sentiment Analysis (ABSA). Our research investigates the performance of fine-tuned LLMs compared to state-of-the-art (SOTA) methods on English and German datasets, particularly in resource-scarce scenarios.

## Key Findings

- Fine-tuned LLMs outperform current SOTA methods in data-scarce scenarios, especially for complex ABSA tasks.

- Performance remains stable even with reduced dataset sizes (as low as 500 or 1,000 samples).

- Concise prompts are generally sufficient when combined with well-optimized hyperparameters.

- New SOTA F1 scores:

  ACSA (82.48) and E2E (81.77) on the Rest-16 dataset

  ACSA (85.45) and TASD (75.13) on the GERestaurant dataset

## Datasets

We use the following datasets in our evaluation:

- Rest-16: English restaurant review dataset from SemEval 2016 ([Pontiki et. al, 2016](https://alt.qcri.org/semeval2016/task5/))

- GERestaurant: German restaurant review dataset ([Hellwig et. al, 2024](https://aclanthology.org/2024.konvens-main.14/))

## Tasks and Methodology

We evaluate our fine-tuned LLMs on four commonly-used ABSA subtasks: Aspect Category Detection (ACD), Aspect Category Sentiment Analysis (ACSA), End-To-End ABSA (E2E), and Target Aspect Sentiment Detection (TASD). Our approach involves instruction fine-tuning LLaMA-3-8B using Quantized Low-Rank Adaptation (QLoRA) to efficiently adapt the model to ABSA tasks. We conduct experiments with varying dataset sizes (500, 1,000, and full dataset) to assess performance under resource constraints. Additionally, we explore the impact of different prompt formulations and hyperparameter tuning.

## Repository Structure
```
📂 root/
├── 📂 data/              # Preprocessed datasets
├── 📂 src/               # Code for our instruction fine-tuned LLM approach
├── 📂 baselines/         # Code for our baseline approaches, scripts for execution as well as files for setup and dataset-preparation
├── 📂 scripts/           # Script files for running the evaluation of our approach, Jupyter-Notebooks for analysis of the results
├── 📂 results/           # In-depth results and condition parameters for each evaluation run 
```
## Reproduction (Tested on Python v3.10)
1. Run the setup.sh file of the approach to install all required packages (e. g. ```/scripts/setup.sh``` or ```/baselines/mvp/setup.sh```)
2. Run the run_APPROACH[_PHASE].py file of the approach to execute the training and evaluation. 

## Citation

If you use our work, please cite:
```
@article{TBD,
  title={TBD},
  author={TBD},
  journal={TBD},
  year={2025}
}
```

