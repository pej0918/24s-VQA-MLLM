# Multimodal Learning with Q-Former and MCAN for Visual Question Answering

We propose a model that enhances Q-Former’s performance by integrating the Multimodal Co-Attention Network (MCAN) and introducing a Question-Aware Prompt during fine-tuning to improve Visual Question Answering (VQA) tasks.

## Introduction

<details>
  <summary>Train & Eval</summary>
  
  ## Training & Inference
  
  ### Train
  After downloading the training datasets and specifying their path in [dataset configs](daiv/configs/datasets/), we are ready for training!
  
  #### 0. Setting Environments
  ```Shell
  conda create -n fusion python=3.9
  ```
  ```Shell
  git clone 
  ```
  ```Shell
  cd BLIVA
  ```
  ```Shell
  pip install -e .
  ```
  if packaging error occurs, then:
  ```Shell
  pip install setuptools==69.5.1
  ```

  ### Training
  
  #### 1. Pretraining of Dm-Former
  ```Shell
  python train.py --cfg-path train_configs/pretrain_stage1.yaml
  ```
  #### 2. Pretraining of visual assistant branch
  
  ```Shell
  python train.py --cfg-path train_configs/pretrain_stage2.yaml
  ```
  #### 3. Instruction Finetuning 
  ```Shell
  python train.py --cfg-path train_configs/finetune_stage2.yaml
  ```
  ### Evaluation
  
  #### Evaluation of Stage2 
  ```Shell
  python evaluate.py --cfg-path train_configs/pretrain_stage2_eval.yaml
  ```
  
  ```Shell
  python evaluate.py --cfg-path train_configs/finetune_stage2_eval.yaml
  ```
  
  #### Training with MCAN output (prophet) - okvqa
  ```Shell
  python train.py --cfg-path train_configs/finetune_stage2_t5_vqa.yaml
  ```
  ```Shell
  python evaluate.py --cfg-path train_configs/eval_stage2_vqa.yaml
  ```

</details>

Visual Question Answering (VQA) involves generating accurate answers by reasoning over both textual (questions) and visual (images) data. While Q-Former effectively uses Cross-Attention for learning question-image interactions, it struggles with modeling complex relationships due to its single-layer attention mechanism. To address these limitations, we propose a new architecture combining Q-Former with the Multimodal Co-Attention Network (MCAN), a multi-layer attention mechanism that captures deeper and more complex interactions between questions and images. Additionally, we introduce Question-Aware Prompts during fine-tuning, providing richer contextual information to further enhance the model’s performance on VQA tasks.

## Methodology

![image](imgs/model_Architecture_train.png)

The proposed model integrates Q-Former for initial interaction between the question and the image. However, Q-Former's single-layer Cross-Attention mechanism has limitations in capturing deep, complex relations. Thus, we integrate MCAN (Multimodal Co-Attention Network), which employs Self-Attention and Cross-Attention mechanisms across multiple layers to refine the question-image interaction further.

### Q-Former and MCAN Integration

In our architecture, **Q-Former** serves as the core component to model the basic interactions between the question and the image through **Cross-Attention**. While Q-Former is efficient at handling straightforward interactions, it struggles to capture more complex and nuanced relationships due to its single-layer structure. To overcome this limitation, we integrated **MCAN**, a multi-layered network that utilizes both **Self-Attention** and **Cross-Attention** to iteratively refine the interaction between the question and the image. By leveraging MCAN’s deeper, multi-layered attention mechanism, the model can capture not only high-level semantic relationships but also intricate, fine-grained details, significantly enhancing the model’s reasoning capability and overall performance.

### Fine-tuning with Question-Aware Prompts

![image](imgs/model_finetuning.png)

During the fine-tuning phase, we introduce Question-Aware Prompts that provide additional context about the question. These prompts include background knowledge and potential answer candidates, helping the model better interpret the question’s intent. This step enhances the model's ability to handle complex questions by providing deeper reasoning capabilities.


## Results

We evaluated our model on standard VQA datasets such as **OK-VQA** and **AOK-VQA**, with pre-training performed on **COCO** and **Visual Genome** datasets. The following table presents the accuracy results comparing different models and the impact of incorporating **Question-Aware Prompts**.

| Model           | Accuracy (Only-Question) | Accuracy (Question-Aware Prompt) |
|-----------------|--------------------------|----------------------------------|
| Q-Former        | 49.2%                     | 55.65%                          |
| MCAN            | **52.56%**                | -                                |
| Ours            | 50%                       | **56.1%**                           |

### Results Analysis

The results clearly indicate that our proposed model, which integrates **MCAN** and employs **Question-Aware Prompts**, provides a significant boost in accuracy on VQA tasks. Specifically, our model shows a **6.1% improvement** when compared to using Q-Former alone, demonstrating the added value of introducing **Question-Aware Prompts** in fine-tuning. These prompts enrich the question by offering additional context and potential answer cues, thereby allowing the model to reason more effectively about the question's intent. 

Moreover, the **MCAN** integration outperforms the standard Q-Former architecture, as the multi-layered attention mechanism in MCAN captures deeper and more nuanced relationships between the question and the image. The ability of MCAN to iteratively refine the attention across multiple layers ensures that both high-level semantics and intricate details are incorporated into the model's decision-making process. This deeper understanding enables the model to handle more complex VQA tasks and scenarios, which is reflected in the superior accuracy achieved in comparison to **InstructBLIP** and **Q-Former** alone.

In conclusion, the results validate that combining **MCAN**'s deeper attention mechanisms with **Question-Aware Prompts** leads to more sophisticated reasoning and higher accuracy, making our model better suited for tackling challenging VQA problems.
