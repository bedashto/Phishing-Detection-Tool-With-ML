# Phishing Detection Tool

This tool uses a machine learning model to detect phishing emails based on content analysis.

## Features
1. **URL Analysis**: Detects suspicious patterns such as IP addresses or phishing keywords.
2. **Email Header Analysis**: Flags anomalies like mismatched 'From' and 'Reply-To' fields.
3. **Machine Learning Model**: Classifies email content as phishing or legitimate.

## Requirements
- Python 3.8+
- scikit-learn
- pandas

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/bedashto/phishing-detector.git
cd phishing-detector
pip install -r requirements.txt
