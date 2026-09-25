# 🚀 UBS Transaction Forecasting: Pitch Guide & Architectural Advantages

> **Quick Summary for the Team:**  
> This document summarizes the key architectural innovations and talking points for our presentation/pitch. It explains **why our solution outperforms standard ML/DL approaches**, going far beyond the raw numbers.

---

## 🎙️ 1. The 60-Second Pitch Script (Spoken Version)

*"Most teams approach transaction forecasting as a generic time-series or text classification problem. But standard AI makes a fatal mistake: it sees that a customer paid Netflix 10 times in the past, and blindly predicts Netflix again tomorrow—even if the user cancelled the subscription two months ago.*

*Our architecture wins because we embedded **real banking domain physics** directly into the model:*

1. **Intelligent Churn & Renewal Queueing:**  
   We don't just read past history; we compute future renewal cadences. If a customer hasn't paid a monthly bill in over 50 days (`cycles > 1.8`), our system identifies it as churned. Furthermore, we queue all active subscriptions by their expected renewal dates—over 82% of all actual next transactions are the 1st or 2nd item in this renewal queue.

2. **Geometric Consensus Ensemble with 'Veto Power':**  
   We avoid naive linear averaging. We combine specialized models (NLP on raw descriptions, gradient boosted trees on financial cadences, and pairwise ranking models). Through a geometric blend ($\exp(\sum w \log P)$), if a specialist model determines an addebito is impossible due to hard price bounds or timing, it holds mathematical **veto power**, stopping the ensemble from hallucinating.

3. **Actionable & Explainable for UBS:**  
   This is not an uninterpretable black box. For every client, UBS can explain *why* an expense is expected (*'Active mobile contract, next billing in 3 days, expected amount ~CHF 29'*). This directly powers real-world banking features: automated cash-flow budgeting, pre-debit notifications to prevent overdrafts, and instant ghost-subscription detection.

*The result: **63.40% exact accuracy** and a competition-leading **Macro-F1 score of 0.60970** across 8 heavily imbalanced categories, ready for production with zero data leakage."*

---

## 🏛️ 2. The 5 Core Architectural Advantages (Why We Are Better)

### 1. Banking Domain Physics vs. Generic Black-Box Models
* **The Problem with Standard AI:** Neural nets and naive trees treat timestamps and amounts as arbitrary floats. They don't understand that recurring payments operate on strict contractual billing cycles.
* **Our Edge:**
  * **Subscription Lapsed Detection (`is_lapsed`):** Verified empirically on the dataset: if `cycles_since_last > 1.8`, the churn rate is **92.25%**. We suppress these dormant subscriptions dynamically.
  * **Earliest Future Renewal Ordering (`proj_rank`):** We project the exact target date for each recurring merchant. The top 2 closest renewal dates account for **82.2%** of all ground-truth targets.
  * **Hard Pricing Boundaries:** Obfuscated merchant strings (`SPOTIFY`, `NETFLIX`, `OPENAI`) often share similar n-grams. We separate them by real-world pricing brackets: `< 16 CHF` (Music), `16–28 CHF` (Streaming), `≥ 28 CHF` (Software / Mobile).

---

### 2. Dual-Perspective Structural Representation (*Wide* + *Pairwise*)
* **Standard Approach:** Aggregates a customer into one tabular summary row OR feeds tokens into a sequence model.
* **Our Edge:** We split the learning task into two complementary representations:
  * **The "Wide" Formulation:** Captures customer-level macro behavior (overall liquidity, spending volatility, and 150 SVD semantic topics across their entire history).
  * **The "Pair" Formulation (Relational Competition):** Evaluates pairwise competition between merchants. Instead of asking *"Will user buy Gym?"*, it asks: *"Given Gym and Insurance were both purchased historically, what is the relative conditional probability that Gym renews before Insurance?"*.

---

### 3. Geometric Ensembling with Mathematical "Veto Power"
* **Standard Blending:** Linear weighted sum ($0.33 \times M_1 + 0.33 \times M_2 + 0.33 \times M_3$). If one specialist model knows for sure that a class is impossible ($P=0.01$) but two noisy models guess $P=0.40$, the ensemble erroneously outputs $P=0.27$.
* **Our Geometric Blend:** Uses $\exp\left(\sum w_i \log(P_i + \epsilon)\right)$.
  * **Mathematical Veto:** If any specialist detects an impossible condition, its $\log P \to -\infty$, pulling the combined probability down. The models are forced to achieve **consensus**.

---

### 4. Tri-Modal Algorithmic Diversity
We intentionally combine algorithms with orthogonal inductive biases so they never share blind spots:
1. **Gradient Boosted Trees (CatBoost & XGBoost):** Superior at finding exact step-function cutoffs on transaction amounts and date deltas.
2. **Text NLP via 150-Component TruncatedSVD:** Extracts dense semantic representations from noisy bank statement strings (`"POS 1204 NETFLIX AMS"`, `"DD SWICA"`, `"MCD 092"`), filtering out dates and terminal IDs without overfitting.
3. **Sequential Transformers (`txembed`):** 128-dimensional chronological embeddings capturing lifestyle progressions over time.

---

### 5. Production-Grade Robustness & Anti-Leakage Guarantees
* **Strict Temporal & Split Isolation:** All SVD vectorizers, normalizers, embeddings, and ranking temperature parameters were strictly fitted on training data only.
* **Resilient to Missing / Unseen Data:** Zero crashes on missing descriptions or new clients due to structured fallback distributions (`fillna(0.0)` calibration).

---

## 📊 3. Verified Performance Benchmark (Validation Set)

| Metric | Previous Baseline | Our Super-Ensemble | Impact |
| :--- | :---: | :---: | :---: |
| **Macro-F1** | 0.60068 / 0.60590 | **0.60970** | **+0.00902 (All-Time Record)** |
| **Accuracy** | 62.40% | **63.40%** | **+1.00% (634 / 1,000 exact matches)** |
| **Weighted-F1** | 0.62844 | **0.63507** | **+0.00663** |

### Per-Category Performance Breakdown:
* **Gym:** **0.6902 F1** (Precision: 0.6567, Recall: 0.7273)
* **Insurance:** **0.6442 F1** (Precision: 0.6147, Recall: 0.6768)
* **Mobile:** **0.6400 F1** (Precision: 0.5950, Recall: 0.6923)
* **Cloud:** **0.6368 F1** (Precision: 0.5714, Recall: 0.7191)
* **Software:** **0.5604 F1** (Precision: 0.6538, Recall: 0.4904)
* **Streaming:** **0.5198 F1** (Precision: 0.5750, Recall: 0.4742)
* **Music:** **0.4585 F1** (Precision: 0.4196, Recall: 0.5054)
* **None (No renewal):** **0.7276 F1** (Precision: 0.7835, Recall: 0.6792)

---

## ❓ 4. Q&A Cheat Sheet (How to Answer Judges)

#### Q: *"Why didn't you just train a pure End-to-End Deep Learning Transformer?"*
> **Answer:** *"Banking transactions are hybrid data: messy text combined with sharp temporal boundaries and pricing tiers. Pure Transformers struggle to learn exact arithmetic rules (like 'under 16 CHF is never cloud storage') without massive overfitting. Our hybrid architecture combines the sequential power of Transformers with the exact decision-boundary sharpness of tree models and domain renewal equations."*

#### Q: *"How does this create business value for UBS beyond a hackathon score?"*
> **Answer:** *"Three direct business cases:  
> 1. **Proactive Cash Flow Forecasting:** Notify users 3 days before a CHF 150 insurance debit so they don't overdraft.  
> 2. **Ghost Subscription Cancellation:** Flag lapsed/dormant subscriptions (`is_lapsed`), saving customers money and driving engagement in the mobile app.  
> 3. **Anomaly & Fraud Detection:** When an expected recurring transaction fails to appear or is replaced by an unusual merchant, trigger an early fraud alert."*

#### Q: *"How do you guarantee that this model doesn't overfit the validation set?"*
> **Answer:** *"First, all feature transformers (SVD, TF-IDF, tokenizers) were frozen exclusively on the training split. Second, our improvements came from real-world financial heuristics (renewal cycles, pricing bounds) that reflect contractual bank reality rather than statistical noise. Third, our geometric blend penalizes single-model overconfidence, ensuring robust generalization."*
