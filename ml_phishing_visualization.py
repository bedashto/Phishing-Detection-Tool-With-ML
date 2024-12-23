import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def train_model_with_visualization(dataset_path):
    """
    Train a phishing detection model and visualize results.
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
    
    # Generate visualizations
    visualize_results(y_test, y_pred)
    
    return model, vectorizer

def visualize_results(y_test, y_pred):
    """
    Generate visualizations for phishing detection model results.
    """
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Phishing"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
    
    # Classification Report
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"], output_dict=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="YlGnBu")
    plt.title("Classification Report")
    plt.show()
    
    # Data Distribution (Optional)
    sns.countplot(x=y_test)
    plt.title("Distribution of Actual Labels")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=["Legitimate", "Phishing"])
    plt.show()

if __name__ == "__main__":
    # Train and visualize
    train_model_with_visualization("emails.csv")
