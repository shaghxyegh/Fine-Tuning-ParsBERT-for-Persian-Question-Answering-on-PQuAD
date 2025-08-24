# Fine-Tuning-ParsBERT-for-Persian-Question-Answering-on- PersianQA
First fine-tuning of Llama-3.1-8B-4bit via API (access rejected) on PersianQA using Colab T4. Fixed dataset errors (RuntimeError: scripts) with JSON upload. Achieved 36.56% EM, 37.43% F1. Challenges: T4 limits, Transformers issues. Next: tune params, no-answer detection.
# Project Report: Fine-Tuning Llama-3.1-8B-4bit via API on PersianQA Dataset

## Project Overview
This was my first attempt to fine-tune a model, following my prompting evaluation and preceding my Persian RAG setup. Using Google Colab’s free T4 GPU, I fine-tuned the `Llama-3.1-8B-4bit` model via an API (due to rejected access on Hugging Face) on the PersianQA dataset for Persian question-answering (QA). PersianQA, similar to SQuAD, contains contexts, questions, and answers (including unanswerable ones). The goal was to adapt the model for accurate Persian QA, handling right-to-left (RTL) text and unanswerable questions.

**Objectives**:
- Load and preprocess PersianQA dataset.
- Fine-tune `Llama-3.1-8B-4bit` via API using Hugging Face’s Transformers.
- Evaluate performance with SQuAD metrics (Exact Match, F1).
- Address Persian text and computational challenges.

**Tech Stack**:
- Python libraries: Transformers (v4.55.3), Datasets, Torch, Evaluate, BitsAndBytes, TRL.
- Model: Llama-3.1-8B-4bit (accessed via API, e.g., Hugging Face Inference API).
- Environment: Google Colab (free T4 GPU, ~15GB VRAM).

## Methodology
1. **Data Preparation**:
   - Loaded PersianQA train (9,000 examples, 2,700 unanswerable) and test (938 examples, 280 unanswerable) from https://huggingface.co/datasets/SajjadAyoubi/persian_qa.
   - Flattened JSON into SQuAD format (`id`, `context`, `question`, `answers`).
   - Tokenized inputs using the model’s tokenizer, mapping answer spans to token positions.
   - Handled unanswerable questions (~30% of test set) by setting default positions to [CLS] token.

2. **Model Fine-Tuning**:
   - Accessed `Llama-3.1-8B-4bit` via API with 4-bit quantization for T4 compatibility.
   - Used `AutoModelForQuestionAnswering` (or equivalent for API) and SFTTrainer from TRL.
   - Training args: 3 epochs, batch size 1, gradient accumulation 8, learning rate 2e-4, BF16, AdamW optimizer (weight decay 0.01).

3. **Evaluation**:
   - Post-processed API logits to extract best answers (top-20 candidates, max length 30 tokens).
   - Computed SQuAD metrics (Exact Match, F1) using `evaluate` library.
   - Verified metrics with dummy data during development.

4. **Saving and Inference**:
   - Saved best model checkpoints via API to `./llama-persianqa-best`.
   - Generated test set predictions and evaluated performance.

## Results
- **Training Performance**: Completed in ~37 minutes on Colab T4 via API. Training loss decreased from 2.33 (Epoch 1) to 0.98 (Epoch 3).
- **Evaluation Metrics** (test set):
  - Exact Match: 36.56%
  - F1 Score: 37.43%
  - Evaluation runtime: ~29 seconds.

## Challenges and Solutions
1. **Model Access Rejection**:
   - **Challenge**: Request to access `Llama-3.1-8B-4bit` on Hugging Face was rejected, likely due to Meta AI restrictions.
   - **Solution**: Used API (e.g., Hugging Face Inference API) for fine-tuning, enabling training without direct model download.

2. **Dataset Loading Error**:
   - **Challenge**: Failed to load PersianQA directly (`RuntimeError: Dataset scripts are no longer supported, but found pquad.py`). JSON access failed due to connectivity issues.
   - **Solution**: Manually downloaded `pqa_train.json` and `pqa_test.json` from https://huggingface.co/datasets/SajjadAyoubi/persian_qa and uploaded to Colab.

3. **Overfitting Risk**:
   - **Challenge**: Small dataset (~10k examples) led to overfitting; model showed memorization signs.
   - **Solution**: Limited to 3 epochs; monitored validation loss (best F1: 37.43% at Epoch 2). Future: Use data augmentation.

4. **Colab T4 GPU Limitations**:
   - **Challenge**: T4’s limited VRAM (~15GB) and timeouts risked OOM errors for 8B model.
   - **Solution**: Used 4-bit quantization via API, batch size 1, gradient accumulation 8. Saved checkpoints to avoid data loss.

5. **Transformers Library Complexity**:
   - **Challenge**: Transformers v4.55.3 had complexities (e.g., renamed `eval_strategy`, deprecated `tokenizer` in Trainer); API integration issues.
   - **Solution**: Adapted to `eval_strategy`, installed compatible `evaluate==0.4.5`. Referenced Hugging Face docs.

## Future Work
- **Hyperparameter Tuning**: Use Optuna to optimize learning rate, batch size, epochs.
- **Data Augmentation**: Apply synonym replacement or back-translation for Persian data.
- **No-Answer Detection**: Add classification head for unanswerable questions.
- **Deployment**: Create API using Hugging Face Inference Endpoints or FastAPI.
- **Resource Upgrade**: Use Colab Pro or A100 GPU for faster training.

## References
- PersianQA Dataset: https://huggingface.co/datasets/SajjadAyoubi/persian_qa
- Hugging Face Docs: https://huggingface.co/docs/transformers
- Model: [Llama-3.1-8B-4bit, e.g., via Hugging Face Inference API]
- BitsAndBytes: https://huggingface.co/docs/bitsandbytes
- TRL: https://huggingface.co/docs/trl
- Hugging Face API: https://huggingface.co/docs/api-inference

**Author**: Shaghayegh Shafiee  
**Date**: August 24, 2025
