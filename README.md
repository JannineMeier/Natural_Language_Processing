# NLP Projects for Lecture

This repository contains the code and documentation for five NLP projects I completed as part of my AI & ML bachelor. Each project focuses on a specific approach to building NLP models, ranging from word embeddings to pretrained large language models (LLMs). Below is an overview of each project:

---

### **Project 1: Word Embeddings + Classifier**
- **Objective**: Train a classifier using precomputed word embeddings.
- **Steps**:
  1. Preprocess data.
  2. Convert sentences into vectors using word embeddings.
  3. Train a classifier on the resulting vectors.
- **Techniques**:
  - Word embeddings: Word2Vec, GloVe, or FastText.

---

### **Project 2: 2-Layer RNN + Single Layer Classifier (Trained Jointly)**
- **Objective**: Train an RNN (LSTM/GRU) and a classifier together.
- **Steps**:
  1. Preprocess data.
  2. Train a 2-layer RNN (LSTM or GRU).
  3. Train a single-layer classifier on top of the RNN output.
- **Techniques**:
  - Recurrent neural networks (LSTM/GRU).

---

### **Project 3: 6-Layer Transformer Encoder + Single Layer Classifier (Trained Jointly)**
- **Objective**: Train a Transformer encoder with a classification layer.
- **Steps**:
  1. Preprocess data.
  2. Train a 6-layer Transformer encoder.
  3. Train a single-layer classifier on top of the encoder.
- **Techniques**:
  - PyTorch's `TransformerEncoder`.

---

### **Project 4: Pretrained Transformer Encoder-Only Model + Single Layer Classifier (Fine-tuned)**
- **Objective**: Fine-tune a pretrained Transformer encoder for classification.
- **Steps**:
  1. Preprocess data.
  2. Fine-tune a pretrained Transformer encoder (e.g., BERT).
  3. Add a single-layer classifier on top and train jointly.
- **Techniques**:
  - Pretrained models: BERT, specifically `BERTForSequenceClassification`.

---

### **Project 5: Pretrained Quantized LLM + PEFT**
- **Objective**: Fine-tune a quantized LLM using PEFT (Parameter-Efficient Fine-Tuning).
- **Steps**:
  1. Preprocess data.
  2. Quantize the pretrained LLM (e.g., LLaMA).
  3. Apply PEFT techniques (e.g., LoRA) to fine-tune the model and classifier.
- **Techniques**:
  - Quantized LLM: Quantized LLaMA using `unsloth`.
  - Fine-tuning: Low-Rank Adaptation (LoRA).

---
