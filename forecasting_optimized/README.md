# 📈 Transaction Forecasting - UBS Swiss AI Weeks

This repository contains an advanced ensemble machine learning pipeline designed to predict a customer's **next recurring purchase category** based on their historical banking transaction data. 

Our pipeline aggressively optimizes for the **Macro-F1 Score** over 8 highly imbalanced target classes, achieving a verified validation score of **0.62963** (Accuracy: **65.20%**, Weighted-F1: **0.65281**).

This README provides a comprehensive, step-by-step breakdown of the architecture, the feature engineering process, the entire "Zoo" of 10+ machine learning models, and the hyperparameter blending strategy that pushes the score to its limit.

---

## 🎯 1. The Challenge & The Data

### The Objective
Given a sequence of historical bank transactions for a specific user, predict the category of the **very next recurring merchant** the user will purchase from. The 8 target categories are:
`['cloud', 'gym', 'insurance', 'mobile', 'music', 'software', 'streaming', 'none']`

### The Imbalance Problem
The dataset is heavily skewed. Categories like `none` (no recurring purchase) or `mobile` might be incredibly common, while `software` or `cloud` are rare. Because the evaluation metric is **Macro-F1**, the model is heavily punished for ignoring rare classes. A naive model that predicts the most common class will score poorly.

### Data Flow Example
A user might have a history like this:
1. `2024-01-01 - 15.00 CHF - "POS PAYMENT SPOTIFY AMSTERDAM"`
2. `2024-01-05 - 50.00 CHF - "DIRECT DEBIT SWICA HEALTH"`
3. `2024-02-01 - 15.00 CHF - "POS PAYMENT SPOTIFY AMSTERDAM"`

*Goal:* The model must recognize the 30-day temporal gap in the "SPOTIFY" transactions and output a high probability for the `music` category.

---

## 🧬 2. Feature Engineering

We transform the raw transaction sequences into two distinct mathematical formats for our models:

### A. The "Wide" Table (Client-Level Features)
We aggregate the user's entire history into a single row containing hundreds of statistical features:
* **Temporal Features:** Average days between transactions, recency of the last transaction, standard deviation of time gaps.
* **Financial Features:** Average transaction amount, total spend, standard deviation of amounts.
* **NLP (Text) Features [The SVD 150-Component Pipeline]:** 
  Bank descriptions are messy. We use `TfidfVectorizer` to count the importance of words across the user's history. Because this creates thousands of columns, we use **TruncatedSVD** to compress these thousands of words into exactly **150 dense mathematical features**. This prevents our tree-based models from overfitting on rare typos while preserving the semantic meaning of the descriptions.

### B. The "Pair" Table (Cross-Merchant Co-occurrence)
Instead of looking at the user as a whole, this table looks at pairs of historical transactions. It asks: *"If a user buys from merchant A, what is the probability they also buy from merchant B?"* This captures complex lifestyle correlations (e.g., people who go to the gym might also buy health insurance).

---

## 🦁 3. The Model Zoo

To achieve a Macro-F1 > 0.60, relying on a single model is impossible. We use an ensemble of **12 distinct models**, divided into three tiers: Base Models, Extra/Contextual Models, and Meta-Learners.

### Tier 1: The Base Models
1. **CatBoost Multiclass:** The workhorse. It reads the "Wide" table (including the 150 SVD text features and time gaps). It is exceptionally good at finding non-linear relationships between the exact dollar amounts, the time delays, and the target category.
2. **XGBoost Pair Model:** It reads the "Pair" table. It ignores the overall user profile and focuses strictly on conditional probabilities of merchant co-occurrences.
3. **Family Model:** A simple but robust statistical baseline. It looks at the user's past categories and simply averages them. If 80% of a user's past recurring payments were `mobile`, it outputs a high probability for `mobile`.
4. **Neural Network (MLP):** A deep learning model trained on the Wide table. It acts as a diversity engine—it often makes different mistakes than the tree-based models, which makes it incredibly valuable for the final blend.

### Tier 2: The Extra / Contextual Models
5. **Ranking Model:** Instead of raw probabilities, this model looks at the absolute frequency rank of the merchants in the user's history.
6. **Description Stream Model:** Treats the user's transactions as a chronological sentence (e.g., `"MCDONALDS -> SPOTIFY -> SWICA -> SPOTIFY"`). It looks for sequential patterns over time.
7. **Embedding Pool Model:** Groups transactions that share similar text embeddings, helping to catch subscriptions that occasionally change their billing names.
8. **Proxy CatBoost & Proxy Pair:** Models trained not on the true target labels, but on "proxy" labels (highly correlated alternative targets). They provide slightly shifted perspectives that stabilize the ensemble.
9. **XGBoost (Standalone):** An alternative tree implementation trained on the Wide table, offering gradient boosting diversity against CatBoost.

### Tier 3: The Specialists & Meta-Learners
10. **The Stacking Meta-Learner:** A model trained *on the outputs* of the other models. It learns when to trust CatBoost and when to trust the Neural Network.
11. **The Micro-Classifier (Digital Goods Specialist):** 
    *The Problem:* The base models confuse `music`, `streaming`, and `software` because they all cost ~10-15 CHF and happen every 30 days.
    *The Solution:* A targeted Logistic Regression model that **only** activates for these specific classes. It uses a `FeatureUnion` of Word N-grams and Character N-grams (e.g., finding the string "flix" inside "ntflix") to aggressively separate digital subscriptions.

---

## 🌪️ 4. The Blender (Ensemble Strategy)

Having 12 models means having 12 arrays of probabilities. Averaging them simply dilutes the strong predictions. We use a **Geometric Blending** strategy optimized via **Random Search**.

### How the Blender Works:
1. **The Base Blend:** We first establish a solid baseline using fixed weights for the Tier 1 models (e.g., 30% CatBoost, 56% Pair Model, 14% Neural Net).
2. **Random Search Loop:** Over 5,000 iterations, a script (`tune_blend_adjustments.py`) randomly injects the extra models (Tier 2 and 3) into the base blend. 
3. **Geometric Averaging:** Instead of `(A + B) / 2`, we use `exp( w * log(A) + (1-w) * log(B) )`. This harshly punishes models that are completely wrong (a probability near 0 drags the whole blend down), forcing the models to agree.
4. **Class-Specific Offsets:** Finally, the blender applies a mathematical bias (offset) to specific classes. For example, if the ensemble is systematically under-predicting the `music` category, the blender might learn to add `+0.70` to the raw logarithmic score of the `music` class across all predictions.

### Verified Validation Performance

The winning ensemble configuration was verified on the 1,000-client validation dataset with the following results:

* **Macro-F1:** **0.62963**
* **Weighted-F1:** **0.65281**
* **Accuracy:** **65.20%** (652 / 1,000 correct)
* **Macro Precision:** **0.62617**
* **Macro Recall:** **0.64140**

#### Detailed Per-Category Breakdown:

| Category | Precision | Recall | F1-Score | Support (Clients) |
| :--- | :---: | :---: | :---: | :---: |
| **`gym`** | 0.6500 | 0.7521 | **0.6973** | 121 |
| **`insurance`** | 0.6765 | 0.6970 | **0.6866** | 99 |
| **`mobile`** | 0.6000 | 0.7500 | **0.6667** | 104 |
| **`cloud`** | 0.6095 | 0.7191 | **0.6598** | 89 |
| **`software`** | 0.6385 | 0.5096 | **0.5668** | 104 |
| **`streaming`** | 0.5882 | 0.5155 | **0.5495** | 97 |
| **`music`** | 0.4434 | 0.5054 | **0.4724** | 93 |
| **`none`** | 0.8032 | 0.6826 | **0.7380** | 293 |
| **Macro Average** | **0.6262** | **0.6414** | **0.6296** | **1,000** |
| **Weighted Average** | **0.6623** | **0.6520** | **0.6528** | **1,000** |
---

## 🚀 5. How to Reproduce the Pipeline

If you want to run this code and beat our score, follow these steps:

1. **Train the SVD Text Features & CatBoost:**
   ```bash
   python train_new_features.py
   ```
   *Tip to beat us: Increase the SVD components from 150 to 300, or swap TF-IDF for pre-trained BERT embeddings.*

2. **Train the Digital Goods Specialist:**
   ```bash
   python micro_classifier_experiment.py
   ```
   *Tip to beat us: Expand the micro-classifier to target `insurance` vs `gym` confusion.*

3. **Optimize the Weights (The Blender):**
   ```bash
   python tune_blend_adjustments.py
   ```
   *Tip to beat us: Run the random search for 50,000 iterations and allow it to search for negative weights.*

4. **Generate the Final Submission:**
   The `predict.py` script reads the JSON configuration generated by the blender and outputs the final probabilities.
   ```bash
   python predict.py
   ```

---
*Developed for the Swiss AI Weeks Hackathon (UBS Challenge)*
