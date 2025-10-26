import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from src.models.baselines import build_resnet18
from src.models.train import WasteDataset


def setup_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    return device, reports_dir


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_data():
    current_dir = Path(__file__).parent
    manifest_path = current_dir.parent.parent / "data" / "unified" / "manifest.csv"
    
    df = pd.read_csv(manifest_path)
    test_dataset = WasteDataset(df, split="test", transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    return test_dataset, test_loader


def load_model(device, num_classes):
    current_dir = Path(__file__).parent
    model_path = current_dir / "baseline.pth"
    
    model = build_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader, device):
    correct, total = 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return np.array(all_labels), np.array(all_preds), test_acc


def save_confusion_matrix(labels, preds, class_names, reports_dir):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    
    plt.figure(figsize=(10, 8))
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_classification_report(labels, preds, class_names, reports_dir):
    report = classification_report(labels, preds, target_names=class_names)
    
    with open(reports_dir / "classification_report.txt", "w") as f:
        f.write(report)


def save_misclassified_examples(labels, preds, test_dataset, class_names, reports_dir, num_examples=5):
    wrong_idx = np.where(labels != preds)[0]
    
    if len(wrong_idx) == 0:
        print("No misclassified images found")
        return

    num_examples = min(num_examples, len(wrong_idx))
    selected_indices = np.random.choice(wrong_idx, size=num_examples, replace=False)
    
    current_dir = Path(__file__).parent
    
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
    if num_examples == 1:
        axes = [axes]
    
    for i, idx in enumerate(selected_indices):
        img_relative_path = test_dataset.df.iloc[idx]['file_path']
        img_path = current_dir.parent.parent / "data" / "raw" / img_relative_path
        
        img = Image.open(img_path).convert("RGB")
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {class_names[labels[idx]]}\nPred: {class_names[preds[idx]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(reports_dir / "misclassified_examples.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_results_summary(accuracy, reports_dir):
    summary = f"Test Results Summary\n{'='*20}\nAccuracy: {accuracy:.4f}\n"
    
    with open(reports_dir / "results_summary.txt", "w") as f:
        f.write(summary)


def main():
    device, reports_dir = setup_environment()
    
    test_dataset, test_loader = load_data()
    class_names = test_dataset.classes
    
    model = load_model(device, len(class_names))
    
    labels, preds, accuracy = evaluate_model(model, test_loader, device)
    
    save_confusion_matrix(labels, preds, class_names, reports_dir)
    save_classification_report(labels, preds, class_names, reports_dir)
    save_misclassified_examples(labels, preds, test_dataset, class_names, reports_dir)
    save_results_summary(accuracy, reports_dir)
    
    print(f"All results saved to '{reports_dir}'")


if __name__ == "__main__":
    main()