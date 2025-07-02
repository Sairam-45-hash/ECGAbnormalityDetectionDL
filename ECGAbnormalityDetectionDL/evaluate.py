
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, dataloader, class_names):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except:
        roc_auc = None

    return report, conf_matrix, roc_auc

def plot_confusion(conf_matrix, class_names, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def print_summary(report_dict, roc_auc):
    print("Classification Report:")
    for cls in report_dict:
        if cls in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        print(f"{cls}: Precision={report_dict[cls]['precision']:.2f}, Recall={report_dict[cls]['recall']:.2f}, F1={report_dict[cls]['f1-score']:.2f}")
    print(f"Overall Accuracy: {report_dict['accuracy']:.4f}")
    print(f"Macro F1-Score: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {report_dict['weighted avg']['f1-score']:.4f}")
    if roc_auc is not None:
        print(f"Multi-class ROC-AUC: {roc_auc:.4f}")
