# 🔍 SentimentLens – Product Review Analyzer

SentimentLens is a premium, high-fidelity Natural Language Processing (NLP) dashboard designed to analyze and classify sentiment within product reviews. Built on top of **Python**, **Scikit-learn**, **NLTK**, and **Streamlit**, it features real-time classification, batch file processing, and interactive metric reports using a midnight glassmorphic UI.

---

## 🌟 Key Features

*   **🏠 Interactive Dashboard**: Monitor key analytics, general positive vs. negative distributions, ratings breakdown, and dynamic word clouds highlighting common review vocabularies.
*   **🔍 Real-Time Classifier**: Paste any review to instantly predict its sentiment polarity (Positive/Negative) with a confidence gauge. SentimentLens will also visually highlight words that positively (🟢) or negatively (🔴) impacted the classification based on model coefficients.
*   **📁 Bulk & Batch Processing**: Upload raw CSV/TXT reviews files, process them all in seconds, preview predictions, and download a complete annotated CSV.
*   **📊 Classifier Performance Stats**: Explore interactive model diagnostics (Confusion Matrix, ROC Curve, and standard Precision/Recall metrics).

---

## 🛠️ Tech Stack & Libraries

*   **Frontend & Dashboard**: [Streamlit](https://streamlit.io/) (with custom CSS injected glassmorphic styling)
*   **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (TF-IDF Vectorizer & Logistic Regression Classifier)
*   **Natural Language Processing**: [NLTK](https://www.nltk.org/) (Tokenization, stopword cleaning)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Visualizations**: [Plotly Express](https://plotly.com/), [Matplotlib](https://matplotlib.org/), [WordCloud](https://github.com/amueller/word_cloud)

---

## 🚀 Getting Started (Step-by-Step)

Follow these simple instructions to set up the project locally on your machine.

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your computer. You can check your Python version by running:
```bash
python --version
```

### 2. Clone the Repository
```bash
git clone https://github.com/Karankumar2403/SentimentLens.git
cd SentimentLens
```

### 3. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Train the Model
Run the training script to generate the synthetic reviews dataset, pre-process the corpus, and train/save the TF-IDF + Logistic Regression model (`model.pkl`):
```bash
python train_model.py
```

### 5. Launch the Dashboard App
Start the Streamlit application:
```bash
streamlit run app.py
```
This will open the application automatically in your browser (usually at `http://localhost:8501`).

---

## 📁 Repository Structure

```
├── app.py                # Main Streamlit dashboard code (pages, styles, visualizations)
├── train_model.py        # Model training script & synthetic reviews dataset generator
├── test_reviews.csv      # Sample review file for batch upload verification
├── requirements.txt      # List of dependencies to install
├── model.pkl             # Trained model artifact (generated after training)
└── README.md             # Project documentation (this file)
```

---

## 🎓 How the Model Works
1.  **Text Preprocessing**: Normalizes text by removing non-alphabetic characters, tokenizing sentences, lowercasing, and filtering out common English stopwords using `NLTK`.
2.  **Vectorization (TF-IDF)**: Converts clean textual review tokens into numerical features using term frequency-inverse document frequency weighting.
3.  **Classification (Logistic Regression)**: Evaluates the TF-IDF weights to predict a binary label: `1` (Positive) or `0` (Negative). It uses prediction probabilities for confidence scoring.

---
Developed by [Karan Kumar](https://github.com/Karankumar2403). Feel free to star ⭐ the repository if you found this project helpful!
