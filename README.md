# SMS Spam Classification

A simple text classification project that detects whether an SMS message is **spam** or **ham** (not spam), built as a practical assessment task.

## Dataset

- **Name:** SMS Spam Collection
- **Size:** 5,572 labeled SMS messages (4,825 ham / 747 spam) — well above the minimum 200-row requirement
- **Source:** [SMS Spam Collection (UCI / mirrored dataset)](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)
- **Why this dataset:** It is a well-known, publicly available, real-world text classification benchmark with a clear binary label, minimal noise, and a realistic class imbalance (similar to real spam-filtering scenarios), making it a good fit for demonstrating an end-to-end NLP classification pipeline.

## Approach

### 1. Preprocessing
- Lowercased all text
- Removed URLs
- Removed numbers
- Removed punctuation
- Normalized extra whitespace

### 2. Train/Test Split
- 80% train / 20% test
- Stratified split to preserve the spam/ham ratio in both sets

### 3. Feature Extraction
- **TF-IDF Vectorization** (`TfidfVectorizer`) with English stop-word removal and a maximum of 5,000 features
- TF-IDF was chosen over simple Bag-of-Words because it down-weights very common words and highlights terms that are more distinctive of spam vs. ham messages

### 4. Model
- **Logistic Regression** (`max_iter=1000`)
- Chosen because it is a strong, fast, and well-understood baseline for text classification on TF-IDF features, performs well on relatively small/medium datasets, and produces probability scores along with easily interpretable coefficients

## 5. Evaluation Metrics

| Class / Metric | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| Ham             | 0.96    | 1.00 | 0.98   | 966     |
| Spam            | 0.99    | 0.73 | 0.84   | 149     |
| **Accuracy**    |         |      | **0.9659** | 1115  |
| Macro Avg       | 0.98    | 0.87 | 0.91   | 1115    |
| Weighted Avg    | 0.96    | 0.96 | 0.96   | 1115    |

**Confusion Matrix:**

| | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 966 | 0 |
| **Actual Spam** | 38 | 111 |

**Interpretation:** The model achieves 96.59% overall accuracy. It is very conservative about flagging spam (precision = 1.00 — never wrongly labels a real message as spam), but misses some spam messages (recall = 0.745). This trade-off is reasonable for a spam filter, since falsely blocking a legitimate message is usually costlier than letting a few spam messages through.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook Spam_Detector.ipynb
```

Run all cells in order. The notebook will:
1. Load and preprocess the dataset (`spam.csv`)
2. Split it into train/test sets
3. Train a TF-IDF + Logistic Regression model
4. Print accuracy, precision, recall, F1-score, classification report, and confusion matrix
5. Run predictions on example messages

## Assumptions

- The dataset file (spam.csv) is included in this repository, so no internet download is required at run-time.
- A fixed `random_state=42` is used for reproducibility of the train/test split.
- "Spam" is treated as the positive class for precision/recall/F1 calculations.

## Project Structure

```
.
├── Spam_Detector.ipynb   # Main notebook: preprocessing, training, evaluation
├── spam.csv              # Dataset (SMS Spam Collection)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```
