import numpy as np
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import torch
from scipy.special import softmax


def validate_classification_results(
    adata,
    cell_type_probs,
    le_cell_type,
):
    cell_type_argmax_probs = cell_type_probs.argmax(1)
    predicted_cell_type = le_cell_type.inverse_transform(cell_type_argmax_probs)

    # confusion matrix
    fig, axes = plt.subplots(figsize=(10, 10))
    
    cm = confusion_matrix(
        adata.obs["cell_type"].values,
        predicted_cell_type,
        labels=le_cell_type.classes_
    )
    cmd = ConfusionMatrixDisplay(
        cm,
        display_labels=le_cell_type.classes_
    )
    cmd.plot(ax=axes)

    y_true = adata.obs["cell_type"].values
    classes = le_cell_type.classes_

    # --- Classification metrics ---
    accuracy = accuracy_score(y_true, predicted_cell_type)
    precision = precision_score(y_true, predicted_cell_type, average="weighted")
    recall = recall_score(y_true, predicted_cell_type, average="weighted")
    f1 = f1_score(y_true, predicted_cell_type, average="weighted")

    report = classification_report(y_true, predicted_cell_type, target_names=classes)
    print("Accuracy:", accuracy)
    print("Precision (weighted):", precision)
    print("Recall (weighted):", recall)
    print("F1-score (weighted):", f1)
    print("\nFull classification report:\n")
    print(report)

    # --- ROC AUC for multiclass ---
    # Binarize labels for one-hot encoding
    y_true_bin = label_binarize(
        [np.where(classes == c)[0][0] for c in y_true], classes=range(len(classes))
    )

    # Compute ROC curve and AUC for each class
    plt.figure(figsize=(10, 8))
    classes_roc_auc = {}
    classes_fpr = {}
    classes_tpr = {}
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], cell_type_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_label} (AUC = {roc_auc:.2f})")
        classes_roc_auc[class_label] = roc_auc
        classes_fpr[class_label] = fpr
        classes_tpr[class_label] = tpr

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Multiclass Cell Type Prediction")
    plt.legend(loc="lower right")
    plt.show()

    # --- Optional: overall macro/micro ROC AUC ---
    macro_roc_auc = roc_auc_score(y_true_bin, cell_type_probs, average="macro")
    micro_roc_auc = roc_auc_score(y_true_bin, cell_type_probs, average="micro")
    print(f"Macro-average ROC AUC: {macro_roc_auc:.3f}")
    print(f"Micro-average ROC AUC: {micro_roc_auc:.3f}")
    return cm, accuracy, precision, recall, f1, report, classes_roc_auc, classes_fpr, classes_tpr, macro_roc_auc, micro_roc_auc