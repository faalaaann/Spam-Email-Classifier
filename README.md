# Spam Email Classifier

A Jupyter Notebook that trains and compares two classic machine-learning spam filters:

- **Multinomial Naive Bayes**
- **Linear Support Vector Machine (SVM)**

The pipeline normalizes email text, extracts TF-IDF word and bigram features, tunes each classifier with 5-fold cross-validation, and evaluates the selected model on a held-out test set. It also supports custom datasets, individual predictions, and model export.

> The built-in dataset is synthetic and template-based. Its metrics are useful for demonstrating the workflow, but real-world performance should be measured with a representative email corpus.

## Features

- URL, email address, money amount, number, capitalization, and punctuation normalization
- TF-IDF features with unigrams and bigrams
- Naive Bayes and Linear SVM model comparison
- Cross-validated hyperparameter selection
- Accuracy, precision, recall, F1, confusion matrix, false-positive, and false-negative reporting
- Support for CSV and tab-separated SMS Spam Collection-style data
- Individual email and text-file prediction
- Export of the best trained model with `joblib`

## Requirements

- Python 3.9+
- NumPy
- pandas
- scikit-learn
- joblib
- Jupyter Notebook or JupyterLab

Install the dependencies:

```bash
pip install numpy pandas scikit-learn joblib jupyter
```

## Getting started

Clone the repository and open the notebook:

```bash
git clone https://github.com/faalaaann/Spam-Email-Classifier.git
cd Spam-Email-Classifier
jupyter notebook "spam classification.ipynb"
```

Run the notebook from top to bottom. With the built-in sample data, the notebook generates 2,000 synthetic emails, compares both classifiers, selects the best model by cross-validated F1, saves it as `spam_model.joblib`, and runs predictions on example messages.

## Using custom data

The notebook accepts either of the following formats.

### CSV

The CSV must contain one label column and one message column. Supported label names include `label`, `v1`, `class`, and `category`; supported message names include `text`, `v2`, `message`, `email`, and `body`.

Labels may be `spam`/`ham` or `1`/`0`.

Example:

```csv
label,text
ham,"Are we still meeting at 10?"
spam,"WIN a free prize now! Click this link."
```

This also supports the common Kaggle `spam.csv` format from the SMS Spam Collection.

### Headerless tab-separated data

The original SMS Spam Collection format is also supported:

```text
ham	Are we still meeting at 10?
spam	WIN a free prize now! Click this link.
```

## Prediction examples

The notebook's implementation provides command-line-style options for the classifier entry point:

```bash
python spam_classifier.py
python spam_classifier.py --data spam.csv
python spam_classifier.py --predict "Subject: hi - you won $1000, click http://example.com"
python spam_classifier.py --predict-file message1.txt message2.txt
python spam_classifier.py --save-sample data/
```

The repository currently stores the implementation in the notebook rather than as a standalone `spam_classifier.py` file. To use these options, run the corresponding code from the notebook or extract the implementation cell into a Python script.

## Model output

The reported metrics treat spam as the positive class:

- **False positive (FP):** legitimate email classified as spam
- **False negative (FN):** spam email classified as legitimate

The trained best model is saved to:

```text
spam_model.joblib
```

## Project structure

```text
.
├── spam classification.ipynb
└── README.md
```

## License

No license has been specified for this repository.
