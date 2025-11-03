# 🎬 IMDB Movie Review Sentiment Analysis

A deep learning-based sentiment analysis application that classifies IMDB movie reviews as positive or negative using a Simple RNN (Recurrent Neural Network) model. The application features an interactive Streamlit web interface for easy user interaction.

## 🌟 Features

- **Real-time Sentiment Analysis**: Instantly classify movie reviews as positive or negative
- **Confidence Scoring**: Get prediction confidence scores for each classification
- **User-Friendly Interface**: Clean and intuitive Streamlit web UI
- **Pre-trained Model**: Uses a pre-trained Simple RNN model trained on IMDB dataset
- **Text Preprocessing**: Automatically handles text preprocessing and tokenization

## 📋 Prerequisites

- Python 3.7 or higher
- pip package manager

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/codezshubham/Movie-Sentiment.git
   cd Movie-Sentiment
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. **Run the Streamlit application**
   ```bash
   streamlit run main.py
   ```

2. **Access the web interface**
   - The application will automatically open in your default web browser
   - Or navigate to `http://localhost:8501`

3. **Analyze movie reviews**
   - Enter your movie review in the text area
   - Click the "🔍 Analyze Sentiment" button
   - View the sentiment classification and confidence score

## 📦 Dependencies

- **tensorflow**: Deep learning framework for model loading and prediction
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **streamlit**: Web application framework

## 🏗️ Technical Details

### Model Architecture
- **Type**: Simple RNN (Recurrent Neural Network)
- **Dataset**: IMDB Movie Reviews (50,000 reviews)
- **Input**: Sequences of up to 500 words
- **Output**: Binary classification (Positive/Negative)
- **Activation**: ReLU activation in hidden layers, sigmoid in output layer

### Text Processing
- Reviews are converted to lowercase
- Words are encoded using IMDB word index
- Sequences are padded to a maximum length of 500 words
- Unknown words are assigned a default index

### Prediction
- Sentiment is classified as:
  - **Positive**: Prediction score > 0.5
  - **Negative**: Prediction score ≤ 0.5

## 📁 Project Structure

```
Movie-Sentiment/
├── main.py                 # Main application file with Streamlit UI
├── simple_rnn_imdb.h5     # Pre-trained RNN model
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🎨 User Interface

The application features a modern, styled interface with:
- Custom color-coded sentiment results (green for positive, red for negative)
- Responsive design
- Clear visual feedback
- Confidence score display

## 🔍 Example Usage

**Sample Review:**
```
"This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
```

**Expected Output:**
- Sentiment: Positive
- Confidence Score: ~0.95

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available for educational and personal use.

## 👤 Author

**Shubham**
- GitHub: [@codezshubham](https://github.com/codezshubham)

## 🙏 Acknowledgments

- IMDB dataset from Keras datasets
- TensorFlow/Keras for the deep learning framework
- Streamlit for the web application framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

Made with ❤️ using Python and TensorFlow
