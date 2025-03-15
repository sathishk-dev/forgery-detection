from flask import Flask, request, render_template
import torch
import cv2
import numpy as np
from dpmsn_model import DPMSN

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DPMSN().to(device)
model.load_state_dict(torch.load("models/dpmsn_model.pth"))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256, 256))
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
        image = image.to(device)

        with torch.no_grad():
            output = model(image).squeeze().cpu().numpy()

        return f"Forgery Detected: {np.sum(output) > 500}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
