import gradio as gr
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import datetime

# Load your trained model
model = load_model("waste_model.h5")

# Define your class labels (order must match training)
class_labels = ["Organic", "Recyclable"]

# Optional: initialize log storage
prediction_log = []

# Prediction function with details
def predict_waste(img):
    # Resize and normalize image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    pred_idx = np.argmax(prediction)
    confidence = float(prediction[pred_idx]) * 100
    label = class_labels[pred_idx]

    # Format prediction bar chart
    prob_dict = {class_labels[i]: float(prediction[i]) for i in range(len(class_labels))}
    
        # Suggestion text
    if label == "Organic":
        suggestion = (
            "✅ Tip: Organic waste like food scraps or leaves can be composted at home.\n"
            "🌱 Consider using a compost bin to turn it into fertilizer.\n"
            "🚫 Avoid mixing it with plastic or metal waste."
        )
    else:  # Recyclable
        suggestion = (
            "♻️ Tip: Recyclable items like bottles, cans, or cardboard should be rinsed and sorted.\n"
            "📦 Flatten boxes before disposal.\n"
            "🚫 Do not put food-contaminated recyclables (like greasy pizza boxes) in the bin."
        )

    # Log prediction
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_log.append({
        "Timestamp": timestamp,
        "Prediction": label,
        "Confidence": f"{confidence:.2f}%",
        "Suggestion": suggestion,
        "Organic": f"{prediction[0]*100:.2f}%",
        "Recyclable": f"{prediction[1]*100:.2f}%"
    })

    return label, confidence, prob_dict, img, suggestion

# Define Gradio interface
interface = gr.Interface(
    fn=predict_waste,
    inputs=gr.Image(type="pil", label="Upload Waste Image"),
    outputs=[
        gr.Text(label="Predicted Class"),
        gr.Number(label="Confidence (%)"),
        gr.Label(label="Prediction Probabilities"),
        gr.Image(label="Uploaded Image"),
        gr.Textbox(label="Suggestion", lines=4),
    ],
    title="♻️ Automated Waste Segregation Assistant",
    description="Upload a waste image to detect if it's Organic or Recyclable. Uses a deep learning model trained with MobileNetV2.",
    theme="default"
)

# Optional: Add a secondary button to export log
def export_log():
    df = pd.DataFrame(prediction_log)
    df.to_csv("prediction_log.csv", index=False)
    return "prediction_log.csv"

download_interface = gr.Interface(fn=export_log, inputs=[], outputs="file", title="📥 Export Prediction Log")

# Combine both interfaces
app = gr.TabbedInterface([interface, download_interface], ["Predict Waste", "Download Logs"])

# Launch it
if __name__ == "__main__":
    print("🚀 Launching Waste Classifier...")
    app.launch(debug=True)
