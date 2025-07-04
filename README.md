# ECGAbnormalityDetectionDL
A deep learning-based system for robust and accurate detection of cardiac abnormalities from ECG signals


# CardioXplainAI

**CardioXplainAI** is an interpretable deep learning framework proposed for automated detection of cardiac abnormalities from ECG signals. This work is associated with the paper:

> **CardioXplainAI: An Interpretable Hybrid TCN-GRU Deep Learning Framework with Adaptive Attention for ECG-Based Cardiac Abnormality Recognition**  
> *Sairam Vallabhuni, Dr. P.V.Naganjaneyulu and Dr. N.Renuka*  
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
├── CardioXplainAI_v1.ipynb    # Jupyter notebook for structured development version 1
├── Notebook V2.ipynb          # Improved/cleaned notebook with modular sections
├── rhythm_model.pth           # (Optional) Saved model checkpoint (if applicable to your extension)
```


## 🧪 How to Run
1. Place your `.mat` ECG files in a folder named `data/`
2. Run `main.py` to execute the complete pipeline
3. Alternatively, open `CardioXplainAI_Demo.ipynb` for an interactive demo

## 📊 Example Results
- Accurcay -97.89 %
- Precision-97.52 %
- Recall-97.36 %
- F1-score- 97.44
- Real-time explanation overlays using attention and Grad-CAM

## 📌 Notes
- This repository supports reproducibility of the proposed TBME submission.
- Designed for researchers in medical AI, digital health, and clinical diagnostics.

## 📄 License & Contact
This code is part of academic research and is under review. For usage or collaboration, please contact the corresponding author(s):

- **Sairam Vallabhuni** – [Research Scholar, Department of ECE,
Mizoram University, Aizawl, Mizoram, India-796004, Email: vallabhuni.sairam@gmail.com]  
- **Dr. P.V.Naganjaneyulu** – [Professor, Department of ECE,
Mizoram University, Aizawl, Mizoram, India-796004, Email: pvnaganjaneyulu@gmail.com]
- **Dr. N.Renuka** – [Professor,Department of ECE, R.V.R & JC College of Engineering, Guntur, Andhra Pradesh India-522019.
Email: nrenuka@rvrjc.ac.in]

---

## 📚 Citation

> Sairam Vallabhuni, Dr. P.V.Naganjaneyulu and Dr. N.Renuka, “CardioXplainAI: An Interpretable Hybrid TCN-GRU Deep Learning Framework with Adaptive Attention for ECG-Based Cardiac Abnormality Recognition,” *IEEE Transactions on Biomedical Engineering (TBME)*, Under Review, 2024.

```bibtex
@article{cardioxplainai2024,
  author    = {Sairam Vallabhuni, Dr. P.V.Naganjaneyulu and Dr. N.Renuka},
  title     = {CardioXplainAI: An Interpretable Hybrid TCN-GRU Deep Learning Framework with Adaptive Attention for ECG-Based Cardiac Abnormality Recognition},
  journal   = {IEEE Transactions on Biomedical Engineering (TBME)},
  year      = {2024},
  note      = {Under Review}
}
```
