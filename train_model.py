import numpy as np
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# 1. Generate Synthetic Product Review Dataset for demonstration & training
def generate_synthetic_data(num_samples=10000):
    np.random.seed(42)
    
    pos_adjectives = ["excellent", "amazing", "great", "awesome", "fantastic", "perfect", "love", "best", "highly recommended", 
                      "beautiful", "superb", "durable", "sturdy", "fast", "reliable", "brilliant", "smooth", "wonderful", "outstanding"]
    neg_adjectives = ["terrible", "awful", "worst", "waste", "useless", "broken", "cheap", "poor", "hate", "bad", "disappointed",
                      "defect", "broken", "slow", "horrible", "refund", "returned", "annoying", "fails", "garbage", "trash"]
    
    nouns = ["product", "item", "purchase", "device", "quality", "service", "experience", "delivery", "packaging", "sound", "battery", "screen"]
    
    pos_templates = [
        "This is an {adj} {noun}! I really {adj_verb} it.",
        "The {noun} is absolutely {adj}. Highly recommend to anyone.",
        "Very {adj} quality, works exactly as described.",
        "I am so happy with this {noun}. It is the {adj} purchase I have made.",
        "Delivery was {adj_delivery} and the product is {adj}!",
        "Sturdy design and {adj} performance. 5 stars.",
        "Had a {adj} experience. The {noun} is {adj}."
    ]
    
    neg_templates = [
        "This is a {adj} {noun}. Do not buy!",
        "The {noun} is {adj}. It broke after two days of use.",
        "Very {adj} quality. Waste of money.",
        "I am extremely disappointed with this {noun}. It is the {adj} ever.",
        "Delivery was {adj_delivery} and the product is {adj}!",
        "Cheap materials and {adj} performance. Avoid this.",
        "Had an {adj} experience. The {noun} is {adj}."
    ]
    
    reviews = []
    sentiments = []
    ratings = []
    
    for _ in range(num_samples // 2):
        # Positive review
        adj1 = np.random.choice(pos_adjectives)
        adj2 = np.random.choice(pos_adjectives)
        noun = np.random.choice(nouns)
        adj_verb = "love" if adj1 != "love" else "enjoy"
        adj_delivery = "fast" if np.random.rand() > 0.3 else "excellent"
        
        template = np.random.choice(pos_templates)
        review = template.format(adj=adj1, noun=noun, adj_verb=adj_verb, adj_delivery=adj_delivery)
        # Add random noise/variations
        if np.random.rand() > 0.5:
            review += f" The packaging was {np.random.choice(pos_adjectives)}."
        
        reviews.append(review)
        sentiments.append(1)  # Positive
        ratings.append(np.random.choice([4, 5]))
        
    for _ in range(num_samples // 2):
        # Negative review
        adj1 = np.random.choice(neg_adjectives)
        adj2 = np.random.choice(neg_adjectives)
        noun = np.random.choice(nouns)
        adj_delivery = "slow" if np.random.rand() > 0.3 else "bad"
        
        template = np.random.choice(neg_templates)
        review = template.format(adj=adj1, noun=noun, adj_delivery=adj_delivery)
        if np.random.rand() > 0.5:
            review += f" I hate the {noun}. It is {np.random.choice(neg_adjectives)}."
            
        reviews.append(review)
        sentiments.append(0)  # Negative
        ratings.append(np.random.choice([1, 2]))
        
    df = pd.DataFrame({
        'review_text': reviews,
        'sentiment': sentiments,
        'rating': ratings
    })
    
    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)
    return df

# 2. Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(cleaned_tokens)

if __name__ == "__main__":
    print("Generating review dataset...")
    df = generate_synthetic_data(12000)
    
    print("Preprocessing review texts...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    
    print("Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Logistic Regression Model...")
    classifier = LogisticRegression(C=1.0, max_iter=1000)
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate model
    y_pred = classifier.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    model_data = {
        'vectorizer': vectorizer,
        'classifier': classifier,
        'accuracy': acc
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
        
    print("Model components successfully saved to 'model.pkl'.")
