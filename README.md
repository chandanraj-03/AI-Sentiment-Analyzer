# 🎬 AI Sentiment Analyzer (Text-Based)

A modern **AI-powered sentiment analysis web app** that analyzes movie reviews using **text input**.

Built with **Transformers (BERT)** and **Gradio UI**, this app predicts whether a review is **Positive or Negative** with confidence scores in real-time.

---

## 🚀 Features

* ⌨️ **Real-time Text Prediction**
* 📊 **Confidence Score Visualization**
* 📜 **Prediction History (Last 5 entries)**
* ⚡ **Fast GPU Inference (CUDA support)**
* 🎨 **Clean & Modern UI (Gradio)**

---

## 🧠 Model Details

* Model: `bert-base-uncased`
* Task: Binary Sentiment Classification
* Dataset: IMDB Movie Reviews
* Accuracy: ~91–93%
* Framework: 🤗 Transformers + PyTorch

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/sentiment-analyzer.git
cd sentiment-analyzer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch transformers datasets gradio
```

---

## ▶️ Run the App

```bash
python app.py
```

Then open the local URL shown in terminal (usually):

```
http://127.0.0.1:7860
```

---

## 📂 Project Structure

```
.
├── my_model/              # Trained model + tokenizer
├── app.py                 # Gradio app (main file)
├── training.py            # Model training script
├── requirements.txt       # Dependencies
└── README.md
```

---

## 📊 Example

**Input:**

```
"This movie was amazing!"
```

**Output:**

```
Positive (Confidence: 97.2%)
```

---

## ⚙️ Tech Stack

* Python 🐍
* PyTorch 🔥
* Hugging Face Transformers 🤗
* Gradio 🎨

---

## 🚀 Future Improvements

* 🌐 Deploy online (Hugging Face Spaces / Render)
* 📈 Advanced analytics dashboard
* 🎨 UI enhancements

---

## 🤝 Contributing

Pull requests are welcome!
If you’d like to improve the project, feel free to fork and submit changes.

---


⭐ If you like this project, consider giving it a star on GitHub!
