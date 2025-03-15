import torch
import torch.optim as optim
from dataset_loader import dataloader
from dpmsn_model import DPMSN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DPMSN().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "models/dpmsn_model.pth")
print("Training Completed!")
