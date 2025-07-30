# 📰 Fake News Detector using Machine Learning

This is a Streamlit web app that detects whether a news article is **Fake** or **Genuine** using a machine learning model trained on real-world data. The app allows users to either paste a news article or provide a URL (optional). It also supports user feedback collection to improve the model in the future.

---

## Features

-  Detects **Fake** or **Real** news articles
-  Displays **confidence scores** for both classes
-  Accepts feedback from users
-  Works entirely within browser — deployed using **Streamlit Community Cloud**

---

##  Model Information

- **Classifier**: Multinomial Naive Bayes  
- **Vectorizer**: TF-IDF with unigrams  
- **Dataset**: Combination of `True.csv` and `Fake.csv`  
- **Balanced Sampling**: Uses equal-sized samples from both classes  
- **Preprocessing**: Lowercasing, punctuation & digit removal, whitespace normalization  




