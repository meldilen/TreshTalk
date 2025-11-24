import os
from pathlib import Path
from src.pipeline.pipeline import create_complete_pipeline
from src.models.train import WasteDataset
from src.models.eval import get_transforms
from torch.utils.data import DataLoader
import pandas as pd
import cv2

def load_data():
    current_dir = Path(__file__).parent
    manifest_path = current_dir.parent.parent / "data" / "unified" / "manifest.csv"
    
    df = pd.read_csv(manifest_path)
    test_dataset = WasteDataset(df, split="test", transform=get_transforms())
    return test_dataset


def evaluate_pipeline_on_test_images(n=10):
    # Initialize pipeline
    pipeline = create_complete_pipeline()

    # Load test dataset
    test_dataset = load_data()

    reports_dir = Path("src/pipeline/reports")
    reports_dir.mkdir(exist_ok=True)

    print(f"\n🔍 Running pipeline on first {n} test images...\n")

    for i in range(min(n, len(test_dataset))):
        sample = test_dataset[i]

        # Dataset should contain the image path
        _, _, image_path = sample

        if image_path is None:
            raise ValueError("Dataset sample must include 'image_path' field")

        print(f"\n[{i+1}/{n}] Processing: {image_path}")

        # Run full detection + classification
        results = pipeline.process_image(image_path)

        # Save results JSON
        output_json = reports_dir / f"result_{i+1}.json"
        pipeline.save_results(results, str(output_json))

        output_img = reports_dir / f"result_{i+1}.jpg"
        annotated_image = pipeline.visualize_results(
            image_path,
            results,
            save_path=str(output_img),
            show=True 
        )
        print(f"🖼 Saved annotated image: {output_img}")

        print(f"📄 Saved: {output_json}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    evaluate_pipeline_on_test_images(10)
