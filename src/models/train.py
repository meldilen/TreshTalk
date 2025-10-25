import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from baselines import build_resnet18
from PIL import Image
from torch.utils.data import Dataset


# custom transforms for VAL(preprocess) and TRAIN(preprocess + augmentations)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# data set class
class WasteDataset(Dataset):
    def __init__(self, df, split="train", transform=None):
        self.df = df[df['split'] == split].reset_index(drop=True) # сортируем по нужной категории 
        self.transform = transform
        self.classes = sorted(self.df['unified_class'].unique())
        self.class2idx = {c: i for i, c in enumerate(self.classes)} # переводим таргет в цифры (0 - 10)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx): # берем из raw, превращаем в rgb, возращаем рещультат и класс
        row = self.df.iloc[idx]

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        data_root = os.path.join(project_root, "data", "raw")
        file_path = os.path.join(data_root, row['file_path'])
        file_path = os.path.normpath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Image not found: {file_path}")
        
        image = Image.open(file_path).convert("RGB")
        label = self.class2idx[row['unified_class']]
        if self.transform:
            image = self.transform(image)
        return image, label



if __name__ == "__main__":
    # Load manifest and create loaders 
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    manifest_path = os.path.join(project_root, "data", "unified", "manifest.csv")
    
    df = pd.read_csv(manifest_path)

    train_dataset = WasteDataset(df, split="train", transform=train_transforms)
    val_dataset = WasteDataset(df, split="val", transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)


    # Model, loss, optimizer
    num_classes = len(df['unified_class'].unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss() # функция потерь
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # обновление весов модели

    # Training loop
    num_epochs = 5
    best_val_acc = 0.0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src", "models")
            save_dir = os.path.abspath(save_dir)
            os.makedirs(save_dir, exist_ok=True)

            torch.save(model.state_dict(), os.path.join(save_dir, "baseline.pth"))


    plt.figure(figsize=(8,5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/accuracy_curve.png")
    plt.show()

