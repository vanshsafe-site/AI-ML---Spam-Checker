import gradio as gr
import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict(msg):
    msg_vec = vectorizer.transform([msg])
    pred = model.predict(msg_vec)

    return "🚨 SPAM" if pred[0] == 1 else "✅ NOT SPAM"

demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="Spam Message Classifier",
    description="Enter a message and AI will detect if it's spam or not."
)

demo.launch()