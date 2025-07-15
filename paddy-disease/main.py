import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Setup the matplot
matplotlib.use("TkAgg")


train_transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor()
])

class PaddyDiseaseDataset(Dataset):

    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = len(self.class_to_idx)
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(class_path, fname)
                    self.samples.append((full_path, self.class_to_idx[class_name]))
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, label

# Split images
root_dir = 'train_images/'
base_dataset = PaddyDiseaseDataset(root_dir=root_dir, transform=val_transform)
indices = list(range(len(base_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(base_dataset, train_idx)
val_dataset = torch.utils.data.Subset(base_dataset, val_idx)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Visualize the dataset
def visualize_dataset(dataset, num_images=6):
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        img, label = dataset[i]
        img_np = img.permute(1, 2, 0).numpy()
        class_name = dataset.dataset.idx_to_class[label]
        plt.subplot(2, 3, i + 1)
        plt.imshow(img_np)
        plt.title(class_name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

visualize_dataset(train_dataset)

resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
for param in resnet.parameters():
    param.requires_grad = False  

num_classes = len(base_dataset.class_to_idx)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet.to(device)

def evaluate(model, data_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds =torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

num_epochs = 3
for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_acc = evaluate(resnet, val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} — Loss: {running_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}%")

torch.save(resnet.state_dict(), "resnet_paddy_pretrained.pth")
