import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


class ModelEvaluator:

    def evaluate_models(self, y_true, model_predictions, labels=None, show_cm=True, show_cm_pct=True):
        results = []
        n_models = len(model_predictions)

        if show_cm:
            fig_counts, axes_counts = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
            if n_models == 1:
                axes_counts = [axes_counts]

        if show_cm_pct:
            fig_pct, axes_pct = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
            if n_models == 1:
                axes_pct = [axes_pct]

        for i, (name, preds) in enumerate(model_predictions.items()):
            acc = accuracy_score(y_true, preds)
            precision = precision_score(y_true, preds, average='macro', zero_division=0)
            recall = recall_score(y_true, preds, average='macro', zero_division=0)
            f1 = f1_score(y_true, preds, average='macro', zero_division=0)

            print(f"\n{'=' * 50}")
            print(f"Model: {name}")
            print(f"{'=' * 50}")
            print(f"Accuracy        : {acc:.4f}")
            print(f"Precision_macro : {precision:.4f}")
            print(f"Recall_macro    : {recall:.4f}")
            print(f"F1_macro        : {f1:.4f}\n")

            if labels is not None:
                print(classification_report(y_true, preds, target_names=labels, zero_division=0))
            else:
                print(classification_report(y_true, preds, zero_division=0))

            cm = confusion_matrix(y_true, preds)

            if show_cm:
                ax = axes_counts[i]
                ax.imshow(cm, interpolation='nearest')
                ax.set_title(f"{name}\nCounts")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")

                if labels is not None:
                    ax.set_xticks(range(len(labels)))
                    ax.set_yticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45)
                    ax.set_yticklabels(labels)

                for r in range(cm.shape[0]):
                    for c in range(cm.shape[1]):
                        ax.text(c, r, cm[r, c], ha="center", va="center")

            if show_cm_pct:
                ax_pct = axes_pct[i]
                cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                cm_pct = np.nan_to_num(cm_pct) * 100

                ax_pct.imshow(cm_pct, interpolation='nearest')
                ax_pct.set_title(f"{name}\nRow %")
                ax_pct.set_xlabel("Predicted")
                ax_pct.set_ylabel("Actual")

                if labels is not None:
                    ax_pct.set_xticks(range(len(labels)))
                    ax_pct.set_yticks(range(len(labels)))
                    ax_pct.set_xticklabels(labels, rotation=45)
                    ax_pct.set_yticklabels(labels)

                for r in range(cm_pct.shape[0]):
                    for c in range(cm_pct.shape[1]):
                        ax_pct.text(c, r, f"{cm_pct[r, c]:.1f}%", ha="center", va="center")

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision_macro": precision,
                "Recall_macro": recall,
                "F1_macro": f1
            })

        if show_cm:
            fig_counts.tight_layout()
            plt.show()

        if show_cm_pct:
            fig_pct.tight_layout()
            plt.show()

        return pd.DataFrame(results)