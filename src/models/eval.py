import torch
import pandas as pd
from baselines import build_resnet18
from train import WasteDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
from PIL import Image
import os


# Prepare test transformations
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize all images to 224x224
    transforms.ToTensor(),                     # Convert images to tensor
    transforms.Normalize([0.485, 0.456, 0.406], # Normalize using ImageNet stats
                         [0.229, 0.224, 0.225])
])

# Load manifest CSV and create test dataset
df = pd.read_csv("../../data/unified/manifest.csv")
test_dataset = WasteDataset(df, split="test", transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

class_names = test_dataset.classes  # Get class names


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_names)
model = build_resnet18(num_classes=num_classes, pretrained=False)
model.load_state_dict(torch.load("baseline.pth", map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode


# Evaluate on test set
correct, total = 0, 0
all_labels, all_preds = [], []

with torch.no_grad():  # Disable gradient computation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get predicted class

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")


# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
plt.figure(figsize=(10,8))
disp.plot(xticks_rotation=45, cmap='Blues')  # Plot confusion matrix
plt.title("Confusion Matrix")
plt.show()


# Classification Report
report = classification_report(all_labels, all_preds, target_names=class_names)
print("\nClassification Report:\n", report)


#  Show examples of misclassified images
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
wrong_idx = np.where(all_labels != all_preds)[0]  # Indices of wrong predictions

if len(wrong_idx) > 0:
    print(f"\nShowing {min(5, len(wrong_idx))} examples of misclassified images:\n")
    for i in np.random.choice(wrong_idx, size=min(5, len(wrong_idx)), replace=False):
        img_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", test_dataset.df.iloc[i]['file_path'])
        img_path = os.path.normpath(img_path)
        img = Image.open(img_path).convert("RGB")
        plt.imshow(img)  # Show image
        plt.title(f"True: {class_names[all_labels[i]]}, Pred: {class_names[all_preds[i]]}")
        plt.axis('off')
        plt.show()
