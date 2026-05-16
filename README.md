# ID3 Decision Tree from Scratch

A ground-up implementation of the ID3 Decision Tree algorithm in Python, built without using any machine learning libraries for the core logic. Tested on the Iris and Breast Cancer datasets.

---

## Overview

This project implements an ID3 Decision Tree classifier entirely from scratch, including entropy calculation, information gain, recursive tree construction, and prediction. The goal was not to get a quick result using `sklearn`, but to genuinely understand how a decision tree thinks.

---

## Motivation

I wanted to go beyond calling `DecisionTreeClassifier()` and actually understand what happens at each step. How does the tree decide which feature to split on? What does entropy really measure in this context? What breaks if you normalise probabilities incorrectly?

Building it from scratch forced me to answer those questions properly.

---

## Features

- Manual train/test split with shuffle and random seed control
- Entropy calculation with epsilon smoothing to handle edge cases
- Information gain computation across all features at each node
- Recursive tree construction with configurable max depth
- Mean-based thresholding for continuous features
- Single-sample and batch prediction
- Evaluated on two datasets: Iris and Breast Cancer

---

## How It Works

1. **Entropy** is calculated for each subset of labels to measure impurity.
2. **Information Gain** is computed for each feature, comparing entropy before and after a split.
3. The feature with the highest gain is chosen as the split point, using the feature mean as the threshold.
4. The tree is built **recursively**, stopping when a node is pure, max depth is reached, or no gain is possible.
5. Predictions traverse the tree from root to leaf, following the threshold at each node.

One notable design decision: probabilities are computed as `counts / counts.sum()` rather than `counts / len(y)`. This ensures correct normalisation within each subset during recursive splitting, which meaningfully improves accuracy.

---

## Tech Stack

- Python 3
- NumPy
- Pandas
- Matplotlib and Seaborn (visualisation)
- Scikit-learn (dataset loading and evaluation metrics only)

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/md2801/id3-decision-tree.git
cd id3-decision-tree

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Run the notebook
jupyter notebook id3_decision_tree.ipynb
```

---

## Results

| Dataset       | Training Accuracy | Test Accuracy |
|---------------|:-----------------:|:-------------:|
| Iris          | 96.67%            | 100.00%       |
| Breast Cancer | 91.21%            | 93.86%        |

---

## Learning Outcomes

- Understood how entropy and information gain drive feature selection in decision trees
- Discovered how incorrect probability normalisation can silently degrade model performance
- Learned the importance of recursive data subsetting, each split must evaluate entropy on its own subset, not the full dataset
- Practiced building a tree data structure and traversing it for inference
- Gained intuition for why max depth matters as a regularisation tool

---

## Future Improvements

- Add Gini impurity as an alternative splitting criterion
- Implement pruning to reduce overfitting on noisier datasets
- Support categorical features natively without mean thresholding
- Visualise the tree structure as a diagram
- Benchmark against `sklearn`'s implementation across multiple datasets
