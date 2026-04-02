# Second Paper — Bangla Cyberbullying Detection

## Working Title

**Bangla Cyberbullying Detection using Classical ML and Transformer Baselines**

---

## Problem Statement

Cyberbullying on social media platforms is a growing concern, especially in Bengali-speaking communities where automated detection tools are virtually nonexistent. This paper benchmarks strong classical ML and transformer-based baselines on Bangla cyberbullying datasets, providing a systematic comparison and error analysis for this underexplored low-resource language.

---

## Possible Contributions

- Benchmark strong classical + transformer baselines on Bangla cyberbullying data
- Compare classical ML vs transformer models systematically
- Analyze failure cases, label ambiguity, and class imbalance effects
- Provide a reproducible pipeline for Bangla text classification

---

## Key Concepts to Understand

- Text classification pipeline (preprocess → featurize → train → evaluate)
- Bangla text preprocessing (handling Unicode, removing noise, stopwords)
- TF-IDF feature extraction
- Classical classifiers: Logistic Regression, SVM, Naive Bayes, Random Forest
- Transformer fine-tuning for sequence classification
- Class imbalance handling (weighted loss, oversampling)
- Precision, Recall, F1-score, confusion matrix interpretation
- Error analysis methodology

---

## Reading List

### Dataset Papers & Sources
- [Bengali cyberbullying detection dataset paper](https://www.sciencedirect.com/science/article/pii/S2352340925009266)
- [Mendeley dataset](https://data.mendeley.com/datasets/sz5558wrd4/2)
- [Kaggle Bangla cyberbullying dataset](https://www.kaggle.com/datasets/moshiurrahmanfaisal/bangla-cyber-bullying-dataset)

### Transformer Fundamentals
- [HuggingFace LLM Course — Ch1.2: NLP overview](https://huggingface.co/learn/llm-course/en/chapter1/2)
- [HuggingFace LLM Course — Ch1.3: What can transformers do?](https://huggingface.co/learn/llm-course/chapter1/3)
- [HuggingFace LLM Course — Ch1.4: How transformers work](https://huggingface.co/learn/llm-course/en/chapter1/4)

### Metrics & Evaluation
- [Google ML Crash Course — Classification](https://developers.google.com/machine-learning/crash-course/classification)
- [Accuracy, Precision, Recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)

### Reproducibility & Writing
- [ACL reproducibility checklist](https://aclanthology.org/attachments/2025.findings-emnlp.1404.checklist.pdf)

---

## Video Resources

### NLP & Embeddings
- [NLP intro](https://youtu.be/fLvJ8VdHLA0?si=QgK5-wWPDIvVBS9o)
- [Tokenization](https://youtu.be/YdreZtH8oWk?si=XnwgJUBV9BCRomva)
- [Vector Embedding (short)](https://youtube.com/shorts/FJtFZwbvkI4?si=_9O9ZUEK5bF_9eHE)
- [Vector Embedding (explained)](https://youtu.be/dN0lsF2cvm4?si=f9oNtbwSHmsMJLSN)

### Classical ML Models
- [Naive Bayes](https://youtu.be/8vv9julkQEA?si=yE7_boraLqzaWoZH)
- [Decision Tree implementation](https://youtu.be/PHxYNGo8NcI?si=lldNKJKZaZGZQx6D)
- [Random Forest basics](https://youtu.be/gkXX4h3qYm4?si=bkULumqFr7zUQObx)
- [Random Forest implementation](https://youtu.be/ok2s1vV9XW0?si=XjXDX7PjKUW_M_ex)
- [SVM basics](https://youtu.be/NDqACjz5j8g?si=niJ3vNsVK_PaUnhL)
- [SVM implementation](https://youtu.be/FB5EdxAGxQg?si=Tye3KLmUG99Em5tF)

### General ML Workflow
- [CampusX 100 Days ML playlist](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH) — watch selectively: train/test split, text preprocessing, metrics, model evaluation
- [FreeCodeCamp Data Science video](https://www.youtube.com/watch?v=r-uOLxNrNk8) — practical preprocessing and workflow sections only

### Sequence Models (background)
- [RNN explained](https://youtu.be/AsNTP8Kwu80?si=HSkM9MOoSNQlpbNT)
- [RNN implementation](https://youtu.be/0_PgWWmauHk?si=xC67ecbKAze2FD13)
- [LSTM](https://youtu.be/b61DPVFX03I?si=2R83brFKALCbcBLZ)

---

## Tech Stack

- Python
- scikit-learn (classical ML)
- HuggingFace Transformers (transformer fine-tuning)
- pandas, numpy, matplotlib

---

## Minimum Model Targets

1. **TF-IDF + Logistic Regression**
2. **TF-IDF + SVM**
3. **TF-IDF + Naive Bayes**
4. **TF-IDF + Random Forest**
5. **One transformer model** — mBERT, XLM-RoBERTa, or a Bangla-specific BERT

---

## Pipeline Steps

1. Choose and download the Bangla cyberbullying dataset
2. Clean: remove duplicates, missing rows, inspect label balance
3. Decide binary vs multiclass (start binary, extend if time permits)
4. Preprocess: Bangla-specific text cleaning, stopword removal
5. Train/validation/test split with fixed seed
6. Train classical baselines (TF-IDF features)
7. Train transformer baseline (fine-tune pre-trained model)
8. Generate comparison table and confusion matrices
9. Error analysis on misclassified samples
10. Write the paper

---

## Paper Draft Structure

1. Title
2. Abstract
3. Introduction (motivation for Bangla cyberbullying detection)
4. Related Work
5. Dataset Description
6. Methods (classical + transformer pipeline)
7. Experimental Setup
8. Results (baseline comparison table)
9. Error Analysis
10. Limitations
11. Conclusion

---

## What NOT to Waste Time On

- Completing entire ML playlists end-to-end
- Advanced theory not needed for implementation
- Trying too many models without documenting results
- Building custom architectures before baselines are solid
- Rewriting code repeatedly without saving experiments
