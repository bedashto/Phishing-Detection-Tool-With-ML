import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_model(dataset_path):
    """
    Train a machine learning model to classify phishing emails.
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Vectorize email content
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['email_content'])
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model, vectorizer

def predict_email_content(model, vectorizer, email_text):
    """
    Predict if the given email content is phishing or legitimate.
    """
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)
    return "Phishing" if prediction[0] == 1 else "Legitimate"

if __name__ == "__main__":
    # Train the model
    model, vectorizer = train_model("emails.csv")
    
    # Test prediction
    sample_email = "Urgent! Your account has been compromised. Verify your identity now."
    print("Prediction for sample email:", predict_email_content(model, vectorizer, sample_email))
