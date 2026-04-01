import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained("./my_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("./my_model")

history = []

def predict_live(text):
    if not text.strip():
        return "Type something...", 0, "No history"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    sentiment = "Positive" if pred == 1 else "Negative"

    history.insert(0, f"{sentiment} ({confidence*100:.1f}%) - {text[:40]}")
    if len(history) > 5:
        history.pop()

    history_text = "\n".join([f"{i+1}. {h}" for i, h in enumerate(history)])

    return sentiment, confidence, history_text

css = """
#title {text-align: center; font-size: 32px; font-weight: bold;}
#box {border-radius: 10px; padding: 10px;}
"""

with gr.Blocks(css=css) as demo:

    gr.Markdown("<div id='title'>🎬 AI Sentiment Analyzer</div>")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=4,
                placeholder="Type your review here...",
                label="Review",
                elem_id="box"
            )

            sentiment_output = gr.Textbox(
                label="Prediction",
                interactive=False
            )

            confidence_bar = gr.Slider(
                minimum=0,
                maximum=1,
                value=0,
                step=0.01,
                label="Confidence",
                interactive=False
            )

        with gr.Column(scale=1):
            history_output = gr.Markdown("### 📜 History\nNo predictions yet")

    text_input.change(
        fn=predict_live,
        inputs=text_input,
        outputs=[sentiment_output, confidence_bar, history_output],
        show_progress=False
    )

if __name__ == "__main__":
    demo.launch()
