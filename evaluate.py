from sklearn.metrics import f1_score
import torch
from dataset_loader import dataloader
from dpmsn_model import DPMSN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DPMSN().to(device)
model.load_state_dict(torch.load("models/dpmsn_model.pth"))
model.eval()

total_f1 = 0.0

with torch.no_grad():
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).float()

        total_f1 += f1_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='binary')

print(f"Average F1 Score: {total_f1/len(dataloader):.4f}")
