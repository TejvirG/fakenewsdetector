# Fake News Detector (Streamlit demo)

This is a small demo project that classifies news headlines as REAL or FAKE.

Files added:
- `data/news_headlines.csv` — sample labeled headlines used for training (toy dataset for demo)
- `train.py` — trains a TF-IDF + Logistic Regression pipeline and saves it to `model/pipeline.pkl`
- `model.py` — helper to load (or auto-train) the pipeline and predict
- `app.py` — Streamlit app. Run with `streamlit run app.py`
- `requirements.txt` — dependencies

Quick setup (Windows PowerShell):

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Notes
- This demo uses a tiny toy dataset included in `data/` for quick local testing. For a production-grade model you should train on a much larger, curated dataset and add robust evaluation and monitoring.
- The first time you run the app it will auto-train a small model if `model/pipeline.pkl` does not exist.

Made by Tejvir and Vishesh

# Fake News Detection using BERT

A transformer-based NLP system for classifying news articles as **Fake** or **Real** using a fine-tuned BERT model.

The project uses the `bert-base-uncased` model with a publicly available fake and real news dataset. The trained model is evaluated using standard classification metrics and tested on custom news inputs.

## Features

* Binary fake news classification
* BERT-based text classification
* Fine-tuning of `bert-base-uncased`
* WordPiece tokenization
* Training and validation monitoring
* Classification report
* Confusion matrix
* ROC curve and AUC
* Precision-Recall curve
* Exploratory data analysis
* Word cloud analysis
* Custom text prediction

## Dataset

The project uses the **Fake and Real News Dataset** from Kaggle.

The dataset contains two CSV files:

```text
Fake.csv
True.csv
```

Labels are assigned as:

```text
0 → Fake News
1 → Real News
```

The two datasets are combined, shuffled, and divided into training, validation, and test sets.

### Dataset Statistics

| Property       | Value      |
| -------------- | ---------- |
| Total samples  | ~44,000    |
| Classes        | Fake, Real |
| Training set   | 64%        |
| Validation set | 16%        |
| Test set       | 20%        |

Dataset:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Technology Stack

| Component          | Technology                |
| ------------------ | ------------------------- |
| Language           | Python                    |
| Deep Learning      | PyTorch                   |
| NLP Model          | BERT                      |
| NLP Framework      | Hugging Face Transformers |
| Tokenization       | BERT WordPiece Tokenizer  |
| Data Processing    | Pandas, NumPy             |
| ML Utilities       | Scikit-learn              |
| Visualization      | Matplotlib                |
| Training Optimizer | AdamW                     |

## Model

The project uses:

```text
bert-base-uncased
```

BERT is a transformer-based language model that processes text bidirectionally, allowing it to capture contextual relationships between words.

For this project, the pre-trained BERT model is fine-tuned for binary classification:

```text
News Article
     │
     ▼
BERT Tokenizer
     │
     ▼
Input IDs + Attention Mask
     │
     ▼
BERT Encoder
     │
     ▼
Classification Layer
     │
     ▼
Fake / Real
```

## Text Processing

The input text is tokenized using the BERT tokenizer.

Each input is:

* Converted into WordPiece tokens
* Converted into input IDs
* Provided with an attention mask
* Padded or truncated to a fixed maximum length

Minimal preprocessing is used because BERT is designed to learn contextual representations directly from text.

## Training

The BERT model is fine-tuned using the following configuration:

| Parameter | Configuration                 |
| --------- | ----------------------------- |
| Model     | `bert-base-uncased`           |
| Optimizer | AdamW                         |
| Loss      | Cross-Entropy Loss            |
| Scheduler | Linear scheduler with warm-up |
| Epochs    | 3                             |
| Task      | Binary classification         |

During training, the model parameters are updated through backpropagation and validation is performed after each epoch.

Training and validation metrics are monitored to evaluate model performance during fine-tuning.

## Evaluation

The trained model is evaluated using multiple metrics:

### Accuracy

Measures the overall percentage of correctly classified articles.

### Precision

Measures the proportion of predicted samples that belong to the correct class.

### Recall

Measures how many samples from a class are correctly identified.

### F1-Score

Combines precision and recall into a single metric.

### Confusion Matrix

Shows the number of correct and incorrect predictions for both Fake and Real classes.

### ROC-AUC

Measures how effectively the model separates Fake and Real news across different classification thresholds.

### Precision-Recall Curve

Provides additional analysis of the model's classification performance across different thresholds.

## Exploratory Data Analysis

EDA is performed before model training to understand the dataset.

The analysis includes:

* Fake vs Real label distribution
* Text length distribution
* Text length comparison between classes
* Word clouds for Fake news
* Word clouds for Real news

These visualizations help identify differences in vocabulary and writing patterns between the two classes.

## Results

The trained BERT model achieved **over 90% test accuracy** on the dataset, with strong precision, recall, and F1-scores reported for both classes.

The confusion matrix showed a relatively small number of incorrect classifications, while the ROC and Precision-Recall curves indicated strong separation between the two classes.

## Custom Prediction

After training, the model can be used to classify custom news text.

Example:

```text
Input:
[News headline or article]

Output:
FAKE
```

or

```text
Input:
[News headline or article]

Output:
REAL
```

This allows the trained model to be tested on news text outside the original test dataset.

## Project Structure

A typical project structure is:

```text
fake-news-detection/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   └── bert_model/
│
├── notebooks/
│   └── fake_news_detection.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── classification_report.txt
│
├── app.py
├── requirements.txt
└── README.md
```

The exact structure may vary depending on the implementation.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place the downloaded files in the project's data directory:

```text
data/
├── Fake.csv
└── True.csv
```

## Training

Run the training notebook or training script to:

1. Load the dataset
2. Combine Fake and Real news samples
3. Assign binary labels
4. Split the dataset
5. Tokenize the text
6. Fine-tune BERT
7. Validate the model after each epoch
8. Evaluate the trained model
9. Generate performance visualizations

## Prediction

After training, provide a news headline or article to the prediction pipeline.

The model returns one of two classes:

```text
FAKE
REAL
```

## Limitations

The model classifies text based on patterns learned from the training dataset. It does not independently verify the factual accuracy of a news article or check external sources.

Predictions can also be affected by differences between the training data and real-world news, changes in writing styles, and unseen topics.

Therefore, the model should be treated as an automated classification system rather than a replacement for professional fact-checking.

## Future Improvements

* Multilingual fake news detection
* Source credibility features
* Real-time web application deployment
* Larger and more diverse datasets
* Improved handling of long articles
* Integration with external fact-checking sources

## References

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**, NAACL-HLT, 2019.

2. Fake and Real News Dataset, Kaggle:
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

3. Hugging Face Transformers Documentation:
   https://huggingface.co/docs/transformers

4. Paszke, A. et al. **PyTorch: An Imperative Style, High-Performance Deep Learning Library**, NeurIPS, 2019.

5. Pedregosa, F. et al. **Scikit-learn: Machine Learning in Python**, Journal of Machine Learning Research, 2011.

## Author

**Tejvir Singh Grewal**

B.Tech Computer Science Engineering,
Thapar Institute of Engineering & Technology

