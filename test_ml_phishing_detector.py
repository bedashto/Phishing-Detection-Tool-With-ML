import unittest
from ml_phishing_detector import train_model, predict_email_content

class TestMLPhishingDetector(unittest.TestCase):
    def setUp(self):
        self.model, self.vectorizer = train_model("emails.csv")
    
    def test_phishing_email(self):
        phishing_email = "Urgent! Verify your identity now to secure your account."
        prediction = predict_email_content(self.model, self.vectorizer, phishing_email)
        self.assertEqual(prediction, "Phishing")
    
    def test_legitimate_email(self):
        legit_email = "Your meeting with Sarah is confirmed for next Monday."
        prediction = predict_email_content(self.model, self.vectorizer, legit_email)
        self.assertEqual(prediction, "Legitimate")

if __name__ == "__main__":
    unittest.main()
