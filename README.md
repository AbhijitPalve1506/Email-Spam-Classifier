# 🛡️ Spam Guard - Email/SMS Spam Classifier

Spam Guard is a Machine Learning-powered web application built with **Streamlit** that identifies whether an email or SMS message is **Spam** or **Not Spam**.  
It processes the user's input text, analyzes it using an ML model trained on real-world datasets, and displays the result with a confidence score.

## 🚀 Features

- **Real-time Spam Detection**: Instantly check if a message is spam or not.
- **Confidence Score**: Displays how confident the model is about the prediction.
- **Interactive User Interface**: Simple, clean, and mobile-friendly UI using Streamlit.
- **Text Preprocessing Visualization**: See how your input text is cleaned and transformed before prediction.
- **Sample Messages**: Quickly test the system with pre-loaded examples.
- **Educational Insights**: Provides safety tips based on the classification result.

## 🛠️ Technology Stack

- **Python 3**
- **Streamlit** (for frontend and backend)
- **Scikit-learn** (for machine learning model)
- **NLTK** (for text preprocessing: tokenization, stopwords removal, stemming)
- **Pandas & Matplotlib** (for visualization)
- **Pickle** (for model serialization)

## 📚 How It Works

1. **Input**: User enters or pastes a message (email/SMS) into the app.
2. **Text Preprocessing**:
   - Lowercasing
   - Tokenization
   - Removal of stopwords and punctuation
   - Stemming (reducing words to their root form)
3. **Vectorization**: Text is converted into numerical format using **TF-IDF Vectorizer**.
4. **Prediction**: Pre-trained ML model (e.g., Naive Bayes) predicts if the message is Spam (1) or Not Spam (0).
5. **Result Display**: Shows the classification and confidence percentage, along with security advice.

## 📦 Project Structure

```
spam_guard/
│
├── app.py                  # Streamlit app
├── model/
│   ├── model.pkl            # Trained machine learning model
│   └── vectorizer.pkl       # TF-IDF vectorizer
├── requirements.txt         # Required Python libraries
└── README.md                # Project documentation (this file)
```

## 🔥 Installation and Running Locally

1. **Clone the repository**:

```bash
git clone https://github.com/AbhijitPalve1506/spam-guard.git
cd spam-guard
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**:

```bash
streamlit run app.py
```

4. Open the provided local URL in your browser (usually `http://localhost:8501`).


## 🧠 Model Training (Overview)

- **Dataset**: Trained on popular spam detection datasets.
- **Preprocessing**: Used NLTK for tokenization, stopwords removal, and stemming.
- **Feature Extraction**: TF-IDF Vectorization.
- **Model**: Trained using Multinomial Naive Bayes classifier (or equivalent lightweight model ideal for text classification).

## ✨ Future Enhancements

- Add support for file upload (bulk email checking).
- Use more advanced models (like BERT or LSTM-based classifiers).
- Integrate phishing URL detection inside messages.
- Add multilingual spam detection.

## 🙏 Acknowledgements

- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)

# 🛡️ Stay Protected with Spam Guard!