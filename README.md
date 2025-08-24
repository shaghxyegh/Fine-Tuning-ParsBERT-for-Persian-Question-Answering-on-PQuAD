# Fine-Tuning-ParsBERT-for-Persian-Question-Answering-on-PQuAD
My first attempt at fine-tuning ParsBERT on PQuAD for Persian QA using Colab's free T4 GPU. Overcame dataset loading errors, overfitting, and limited VRAM. Achieved 36.56% Exact Match and 37.43% F1 score. Challenges included Transformers library complexities and manual JSON uploads. Future: tune hyperparameters, add no-answer detection.
# Project Report: Fine-Tuning ParsBERT for Persian Question Answering on PQuAD

## Project Overview
This project marks my first attempt at fine-tuning a machine learning model. I used Google Colab’s free tier with a T4 GPU to fine-tune the ParsBERT model (`HooshvareLab/bert-base-parsbert-uncased`) on the PQuAD dataset for Persian question-answering (QA). PQuAD is a Persian dataset similar to SQuAD, containing contexts, questions, and answers (including unanswerable ones). The goal was to adapt a pre-trained language model to accurately answer questions in Persian, handling challenges like right-to-left (RTL) text and unanswerable questions.

**Objectives**:
- Load and preprocess the PQuAD dataset.
- Fine-tune ParsBERT using Hugging Face’s Transformers library.
- Evaluate performance using SQuAD metrics (Exact Match and F1 score).
- Address challenges with Persian text and computational constraints.

**Tech Stack**:
- Python libraries: Transformers (v4.55.3), Datasets, Torch, Evaluate.
- Model: HooshvareLab/bert-base-parsbert-uncased.
- Environment: Google Colab (free tier, T4 GPU).

## Methodology
1. **Data Preparation**:
   - Loaded train (9008 examples) and test (930 examples) JSON files from PQuAD.
   - Flattened nested JSON structure into a SQuAD-like format (`id`, `context`, `question`, `answers`).
   - Tokenized inputs using ParsBERT’s tokenizer, mapping answer spans to token positions.
   - Handled unanswerable questions (~30% of test set) by setting default positions to the [CLS] token.

2. **Model Fine-Tuning**:
   - Used `AutoModelForQuestionAnswering` and `AutoTokenizer` from Transformers.
   - Training args: 3 epochs, batch size 16, learning rate 3e-5, AdamW optimizer with weight decay 0.01.
   - Employed Hugging Face’s Trainer for training and evaluation.

3. **Evaluation**:
   - Post-processed model logits to extract the best answer (top-20 candidates, max answer length 30 tokens).
   - Computed SQuAD metrics (Exact Match, F1) using the `evaluate` library.
   - Verified metric computation with dummy data during development.

4. **Saving and Inference**:
   - Saved the best model based on F1 score to `./parsbert-pquad-best`.
   - Generated predictions on the test set and evaluated performance.

## Results
- **Training Performance**: Completed in ~37 minutes on Colab’s T4 GPU. Training loss decreased from 2.33 (Epoch 1) to 0.98 (Epoch 3).
- **Evaluation Metrics** (test set):
  - Exact Match: 36.56%
  - F1 Score: 37.43%
  - Evaluation runtime: ~29 seconds.


## Challenges and Solutions
1. **Dataset Loading Error**:
   - **Challenge**: Attempting to load PQuAD directly resulted in a `RuntimeError: Dataset scripts are no longer supported, but found pquad.py`. Accessing JSON files online also failed due to connectivity or permission issues.
   - **Solution**: Downloaded the `pqa_train.json` and `pqa_test.json` files manually and uploaded them to Colab for direct loading using `load_dataset("json", ...)`. This bypassed the script error and ensured data accessibility.

2. **Overfitting Risk**:
   - **Challenge**: The relatively small PQuAD dataset (~10k examples) led to overfitting risks. The model showed signs of memorizing training data, reducing generalization.
   - **Solution**: Adjusted the number of epochs to 3 to balance learning and overfitting. Monitored validation loss to ensure stability (best F1: 37.43% at Epoch 2). Future work: Explore data augmentation or dropout.

3. **Colab T4 GPU Limitations**:
   - **Challenge**: Running on Colab’s free T4 GPU with limited VRAM (15GB) and occasional session timeouts constrained training. Large batch sizes or sequence lengths risked out-of-memory (OOM) errors.
   - **Solution**: Used a conservative batch size (16) and max sequence length (384). Monitored memory usage and saved checkpoints to avoid data loss. Future: Upgrade to Colab Pro or use a local GPU.

4. **Transformers Library Complexity**:
   - **Challenge**: The latest Transformers version (4.55.3) introduced complexities, such as deprecated `tokenizer` in Trainer and renamed evaluation strategies (`eval_strategy` instead of `evaluation_strategy`). Compatibility issues with other libraries (e.g., `evaluate`) caused errors.
   - **Solution**: Adapted to `eval_strategy` and ensured compatible library versions (e.g., installed `evaluate==0.4.5`). Referenced Hugging Face documentation to resolve deprecation warnings. Future: Pin specific library versions for stability.

## Future Work
- **Hyperparameter Tuning**: Use tools like Optuna to optimize learning rate, batch size, and epochs.
- **Data Augmentation**: Apply techniques like synonym replacement or back-translation for Persian to increase dataset size.
- **No-Answer Detection**: Add a classification head to better handle unanswerable questions.
- **Deployment**: Create an API using Hugging Face Inference Endpoints or FastAPI.
- **Environment Upgrade**: Use Colab Pro or a local high-performance GPU for faster training.

## References
- PQuAD Dataset: https://huggingface.co/datasets/Gholamreza/pquad.
- Hugging Face Documentation: [Transformers](https://huggingface.co/docs/transformers), [Datasets](https://huggingface.co/docs/datasets).
- Model: [HooshvareLab/bert-base-parsbert-uncased](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased).

**Author**: Shaghayegh Shafiee  
**Date**: August 24, 2025
