# ID3 Decision Tree — Built from Scratch

## Overview

This project is a ground-up implementation of the **ID3 (Iterative Dichotomiser 3) Decision Tree algorithm**, written entirely in Python without relying on scikit-learn's tree models. The algorithm is trained and evaluated on two classic datasets — the **Iris** and **Breast Cancer** datasets — to validate correctness and explore how the model generalises across different problem types.

Everything from entropy calculation and information gain to recursive tree construction and prediction is hand-rolled. No black boxes.

---

## Motivation

It's easy to call `DecisionTreeClassifier().fit(X, y)` and move on. But I wanted to actually understand what's happening underneath — why a particular feature gets chosen to split on, how entropy drives that decision, and what it means for a node to be "pure."

Building this from scratch forced me to think carefully about each step: how probabilities are computed on subsets (not the whole dataset), why that distinction matters for correct entropy values, and how recursive tree construction terminates. The goal was intuition, not just accuracy.

---

## Features

- **Custom entropy calculation** — computed correctly on data subsets, not the full dataset (with a documented explanation of why this matters)
- **Information gain** — selects the best feature at each node by measuring reduction in entropy
- **Recursive tree construction** — builds the tree depth-first with configurable `max_depth` to control overfitting
- **Mean-based thresholding** — handles continuous/numerical features by splitting around the feature mean
- **Manual train/test split** — implemented without `sklearn.model_selection`, using NumPy shuffling and index slicing
- **Prediction traversal** — single-sample and batch prediction via recursive tree traversal
- **Model evaluation** — accuracy scores, classification report, and confusion matrix heatmaps via Seaborn
- **Tested on two datasets** — Iris (multi-class) and Breast Cancer (binary classification)

---

## How It Works

The ID3 algorithm builds a decision tree by greedily selecting the feature that provides the highest **information gain** at each node.

1. **Entropy** measures the disorder/uncertainty in a set of labels. A pure node (all one class) has entropy of 0.
2. **Information Gain** is the reduction in entropy after splitting on a feature — the feature with the highest gain is chosen.
3. **Tree construction** is recursive: split the data, recurse into each branch, and stop when a node is pure, max depth is reached, or no gain remains.
4. **Prediction** traverses the tree from root to leaf, following left/right branches based on whether a feature value is ≤ or > the split threshold.

One key insight documented in the notebook: using `counts / counts.sum()` (subset-aware normalisation) instead of `counts / len(y)` (global dataset size) is critical for correct entropy — the latter can produce probabilities that don't sum to 1 for a given subset, leading to misleading splits and lower accuracy.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| NumPy | Array operations, index manipulation |
| Pandas | Dataset handling and slicing |
| Matplotlib / Seaborn | Visualisation (confusion matrices) |
| scikit-learn | Dataset loading + evaluation metrics only |
| Google Colab | Development environment |

> scikit-learn is used **only** for loading datasets (`load_iris`, `load_breast_cancer`) and computing evaluation metrics (`accuracy_score`, `classification_report`, `confusion_matrix`). The tree algorithm itself is fully custom.

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/id3-decision-tree.git
   cd id3-decision-tree
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook ML-ID3.ipynb
   ```
   Or open directly in [Google Colab](https://colab.research.google.com/).

4. **Run all cells top to bottom** — each section is clearly labelled and builds on the previous one.

---

## Example Output

```
Model Evaluation (Iris Dataset):
Training Accuracy: 100.00%
Testing Accuracy:  96.67%

Classification Report:
              precision    recall  f1-score
  setosa          1.00      1.00      1.00
  versicolor      0.94      0.94      0.94
  virginica       0.94      0.94      0.94
```

*A confusion matrix heatmap is also generated for both datasets.*

---

## Learning Outcomes

- Understood why **entropy must be computed on subsets**, not the global dataset — a subtle but impactful distinction
- Gained intuition for how **information gain drives feature selection** and why greedy splitting works well in practice
- Learned how **recursive algorithms terminate** cleanly using base cases (pure nodes, max depth, zero gain)
- Appreciated the difference between **nominal and continuous features** and why thresholding is needed for numerical data
- Saw firsthand how **max_depth acts as a regulariser** — limiting depth reduces training accuracy slightly but helps generalisation

---

## Future Improvements

- [ ] Add support for **Gini impurity** as an alternative splitting criterion
- [ ] Implement **pruning** (pre- or post-pruning) to improve generalisation on noisier datasets
- [ ] Visualise the tree structure as a proper diagram
- [ ] Extend to handle **missing values** gracefully
- [ ] Benchmark against scikit-learn's `DecisionTreeClassifier` more formally across multiple datasets and splits
