# ECGAbnormalityDetectionDL
A deep learning-based system for robust and accurate detection of cardiac abnormalities from ECG signals


# CardioXplainAI

**CardioXplainAI** is an interpretable deep learning framework proposed for automated detection of cardiac abnormalities from ECG signals. This work is associated with the paper:

> **CardioXplainAI: An Interpretable Hybrid TCN-GRU Deep Learning Framework with Adaptive Attention for ECG-Based Cardiac Abnormality Recognition**  
> *Sairam Vallabhuni, Professor P.V.Naganjaneyulu*  
> *Submitted to IEEE Transactions on Biomedical Engineering (TBME), under review*

## 🔬 Overview

CardioXplainAI integrates a **Temporal Convolutional Network (TCN)** and **Bidirectional GRU (Bi-GRU)** with an **adaptive attention fusion** layer. This hybrid approach captures both fine-grained waveform morphology and sequential rhythm patterns in ECG signals. To ensure clinical trust and transparency, the model incorporates **explainability** using both attention weights and Grad-CAM visualizations.

## 🚀 Features
- ECG signal segmentation, denoising, and normalization
- Hybrid model: TCN for multi-scale temporal patterns + Bi-GRU for sequential dynamics
- Adaptive attention mechanism for feature fusion
- Cross-validated training strategy with early stopping
- Explainability using attention overlays and Grad-CAM
- Performance evaluation using F1-score, accuracy, AUC, and confusion matrix

## 🗂️ Repository Structure
```
├── preprocess.py              # ECG preprocessing and heartbeat segmentation
├── model.py                   # RhythmTCN-GRUAttNet architecture
├── train.py                   # Training loop with optimizer and scheduler
├── evaluate.py                # Evaluation metrics and confusion matrix
├── explainability.py          # Grad-CAM and attention visualization
├── main.py                    # One-click full pipeline execution
├── CardioXplainAI_Demo.ipynb  # Jupyter notebook for interactive walkthrough
```

## 🧪 How to Run
1. Place your `.mat` ECG files in a folder named `data/`
2. Run `main.py` to execute the complete pipeline
3. Alternatively, open `CardioXplainAI_Demo.ipynb` for an interactive demo

## 📊 Example Results
- Macro F1-score: ~93%
- Multi-class ROC-AUC: ~0.95
- Real-time explanation overlays using attention and Grad-CAM

## 📌 Notes
- This repository supports reproducibility of the proposed TBME submission.
- Designed for researchers in medical AI, digital health, and clinical diagnostics.

## 📄 License & Contact
This code is part of academic research and is under review. For usage or collaboration, please contact the corresponding author(s):

- **Sai Ram V.** – [Your Email or GitHub link]  
- **Professor Naganjaneyulu** – [Institution or Department info]

---

## 📚 Citation

> Sai Ram V., and Prof. Naganjaneyulu, “CardioXplainAI: An Interpretable Hybrid TCN-GRU Deep Learning Framework with Adaptive Attention for ECG-Based Cardiac Abnormality Recognition,” *IEEE Transactions on Biomedical Engineering (TBME)*, Under Review, 2024.

```bibtex
@article{cardioxplainai2024,
  author    = {Sai Ram V. and Naganjaneyulu},
  title     = {CardioXplainAI: An Interpretable Hybrid TCN-GRU Deep Learning Framework with Adaptive Attention for ECG-Based Cardiac Abnormality Recognition},
  journal   = {IEEE Transactions on Biomedical Engineering (TBME)},
  year      = {2024},
  note      = {Under Review}
}
```
