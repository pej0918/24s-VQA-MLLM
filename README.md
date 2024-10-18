# Enhancing Q-Former for Visual Question Answering with Multi-layer Co-Attention and Question-Aware Prompts

We propose a model that enhances Q-Former’s performance by integrating the **Modular Co-Attention Network (MCAN)** and introducing a **Question-Aware Prompt** during fine-tuning, improving Visual Question Answering (VQA) tasks.

## Model Architecture Overview
![image](imgs/model_Architecture_train.png)
Visual Question Answering (VQA) involves generating accurate answers by reasoning over both textual (questions) and visual (images) data. While **Q-Former** effectively models question-image interactions through Cross-Attention, it struggles with complex relationships due to its single-layer attention. To address this, we combine Q-Former with the **Modular Co-Attention Network (MCAN)**, introducing a multi-layer attention mechanism for deeper interactions. Additionally, **Question-Aware Prompts** during fine-tuning provide richer contextual information to further boost performance.

### Q-Former and MCAN Integration

In our architecture, **Q-Former** serves as the base model to process interactions between questions and images through its **Cross-Attention** mechanism. However, its single-layer structure has limitations in capturing more complex, nuanced relationships.

To address this, we integrated **MCAN**, a multi-layered network that employs both **Self-Attention** and **Cross-Attention** to progressively refine question-image interactions.

This integration enables the model to extract high-level semantic relationships while also capturing detailed information, which significantly improves the model’s overall reasoning capability.

### Fine-tuning with Question-Aware Prompts

![image](imgs/model_finetuning.png)

During fine-tuning, we introduce **Question-Aware Prompts** to further enhance the model’s performance. These prompts provide additional context about the question, such as background knowledge and potential answer candidates. 

By incorporating these prompts, the model can better interpret the question's intent, allowing for deeper reasoning and more accurate answers. This approach is especially beneficial for complex questions, where the added context enables the model to generate more informed and precise responses. The combination of MCAN’s multi-layered attention mechanism and the use of Question-Aware Prompts during fine-tuning significantly improves the model’s ability to handle challenging VQA tasks.

## Experiment Results

### 1. Environment Setup
```bash
conda create -n fusion python=3.9
```

### 2. Dataset Preparation
Download COCO and Visual Genome datasets, and specify their path in [dataset configs](daiv/configs/datasets/).

### 3. Training the Model
```bash
python train.py --cfg-path train_configs/pretrain_stage1.yaml
```

### 4. Fine-tuning with Question-Aware Prompts
```bash
python train.py --cfg-path train_configs/finetune_stage2.yaml
```

### 5. Evaluation
```bash
python evaluate.py --cfg-path train_configs/finetune_stage2_eval.yaml
```

Here’s an expanded version of the **Results on VQA Datasets** section:

---

### Results on VQA Datasets

We evaluated our model on the **OK-VQA** and **AOK-VQA** datasets, using **COCO** and **Visual Genome** for pre-training. The table below compares the baseline **Q-Former**, **MCAN**, and our enhanced model with and without **Question-Aware Prompts**.

| Model           | Accuracy (Only-Question) | Accuracy (Question-Aware Prompt) |
|-----------------|--------------------------|----------------------------------|
| Q-Former        | 49.2%                     | 55.65%                          |
| MCAN            | **52.56%**                | -                                |
| Ours            | 50%                       | **56.1%**                        |

Our enhanced model, which integrates **MCAN** and **Question-Aware Prompts**, achieved a **6.1% accuracy improvement** when using the prompts compared to the baseline Q-Former. This demonstrates that **Question-Aware Prompts** provide valuable context, enabling the model to better interpret the question’s intent and make more informed predictions. Moreover, **MCAN’s** multi-layer attention mechanism consistently outperformed the single-layer **Q-Former**, especially for complex questions requiring deeper reasoning. These results validate the effectiveness of our approach in improving VQA performance.

## Conclusion

The integration of **MCAN** and **Question-Aware Prompts** enables deeper reasoning, leading to more accurate results for complex VQA tasks. Our model demonstrates significant improvements in accuracy, making it better suited for challenging VQA problems.
