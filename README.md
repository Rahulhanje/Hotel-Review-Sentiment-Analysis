# 🏨 Hotel Review Sentiment Analysis

A machine learning project that classifies hotel reviews into **Negative**, **Neutral**, or **Positive** sentiments using both traditional ML (TF-IDF + Logistic Regression) and state-of-the-art transformer models (DistilBERT). Includes an interactive Streamlit web application for real-time sentiment prediction.

## 🌟 Features

- **Multiple Model Support**: Choose between scikit-learn pipeline or fine-tuned DistilBERT
- **Interactive Web App**: Streamlit-based interface for instant sentiment prediction
- **Comprehensive Evaluation**: Generates detailed classification reports and confusion matrices
- **Easy to Use**: Simple command-line interface with minimal setup

## 📊 Dataset

The project uses hotel review data with two key columns:
- **Review**: Text content of the review
- **Rating**: Numerical rating (1-5 stars)

### Sentiment Label Mapping
- **Ratings 1-2** → Negative 😞
- **Rating 3** → Neutral 😐
- **Ratings 4-5** → Positive 😊

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Rahulhanje/Hotel-Review-Sentiment-Analysis.git
cd Hotel-Review-Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt
```

### Training Models

#### Option 1: Traditional ML Model (TF-IDF + Logistic Regression)
```bash
python train.py
```
- ✅ Faster training
- ✅ Lower computational requirements
- 📁 Model saved to `artifacts/hotel_sentiment_pipeline.joblib`

#### Option 2: DistilBERT Transformer Model
```bash
python train_distilbert.py --csv hotel_reviews.csv --model-name distilbert-base-uncased --epochs 3 --batch-size 16
```
- ✅ Higher accuracy
- ✅ Better understanding of context
- 📁 Model saved to `artifacts/hotel_sentiment_distilbert/`

Both training methods generate evaluation reports in the `reports/` directory:
- `classification_report.txt` - Precision, Recall, F1-Score metrics
- `confusion_matrix.csv` - Confusion matrix for error analysis

### Running the Web Application

```bash
streamlit run app.py
```

The app will automatically detect and load the best available model (DistilBERT if available, otherwise scikit-learn pipeline).

## 📁 Project Structure

```
├── app.py                          # Streamlit web application
├── train.py                        # Traditional ML model training
├── train_distilbert.py             # DistilBERT model training
├── hotel_reviews.csv               # Main dataset
├── hotel_reviews_small.csv         # Sample dataset for testing
├── requirements.txt                # Python dependencies
├── reports/                        # Model evaluation reports
│   ├── classification_report.txt
│   └── confusion_matrix.csv
└── artifacts/                      # Saved models (generated after training)
    ├── hotel_sentiment_pipeline.joblib
    └── hotel_sentiment_distilbert/
```

## 🛠️ Technologies Used

- **Python 3.x**
- **Machine Learning**: scikit-learn, transformers (Hugging Face)
- **Deep Learning**: PyTorch, TensorFlow
- **NLP**: NLTK, DistilBERT
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy

## 📈 Model Performance

The models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open-source and available under the MIT License.

## 👨‍💻 Author

**Rahul Hanje**
- GitHub: [@Rahulhanje](https://github.com/Rahulhanje)

## 🙏 Acknowledgments

- Hotel review dataset contributors
- Hugging Face for transformer models
- Streamlit for the amazing web framework

---

⭐ If you find this project helpful, please consider giving it a star!


