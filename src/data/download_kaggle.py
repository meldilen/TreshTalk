import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil

DATASETS = {
    "trashnet": "feyzazkefe/trashnet",
    "WaRP": "parohod/warp-waste-recycling-plant-dataset",
    "12classes": "mostafaabla/garbage-classification",
    "garbage_classification_1": "asdasdasasdas/garbage-classification",
    "trash_type": "farzadnekouei/trash-type-image-dataset",
    "realwaste": "joebeachcapital/realwaste",
    "garbage_classification_2": "zlatan599/garbage-dataset-classification",
    "garbage_dataset": "sumn2u/garbage-classification-v2"
}

SCRIPT_DIR = Path(__file__).parent.parent.parent
RAW_DIR = SCRIPT_DIR / "data" / "raw"


DATASET_CONFIGS = {
    "garbage_classification_1": {
        "source_path": "Garbage classification/Garbage classification",
        "classes": ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
        "remove_files": ["*.txt"]
    },
    "garbage_classification_2": {
        "source_path": "Garbage_Dataset_Classification/images",
        "classes": ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
        "remove_files": ["*.csv", "*.json", "metadata.*"]
    },
    "realwaste": {
        "source_path": "realwaste-main/RealWaste",
        "classes": ['Cardboard', 'Food Organics', 'Glass', 'Metal',
                    'Miscellaneous Trash', 'Paper', 'Plastic',
                    'Textile Trash', 'Vegetation'],
        "remove_files": ["*.md", "*.txt", "README*", "LICENSE*"]
    }
}


def merge_warp_c_folders():
    """Merge test_crops and train_crops folders in Warp-C dataset"""
    warp_dir = RAW_DIR / "WaRP"

    if not warp_dir.exists():
        print("WaRP folder not found!")
        return

    warp_c_dir = warp_dir / "Warp-C"
    if not warp_c_dir.exists():
        print("Warp-C folder not found!")
        return

    test_crops_dir = warp_c_dir / "test_crops"
    train_crops_dir = warp_c_dir / "train_crops"

    if not test_crops_dir.exists() or not train_crops_dir.exists():
        print("test_crops or train_crops folders not found!")
        return

    merged_dir = warp_c_dir / "merged_crops"
    merged_dir.mkdir(exist_ok=True)

    for category in test_crops_dir.iterdir():
        if not category.is_dir():
            continue

        print(f"📁 Processing category: {category.name}")

        train_category_dir = train_crops_dir / category.name

        if not train_category_dir.exists():
            print(f"⚠️ Category {category.name} not found in train_crops, skipping")
            continue

        for subfolder in category.iterdir():
            if not subfolder.is_dir():
                continue

            subfolder_name = subfolder.name
            train_subfolder = train_category_dir / subfolder_name

            if not train_subfolder.exists():
                print(f"⚠️ Subfolder {subfolder_name} not found in train, skipping")
                continue

            target_dir = merged_dir / subfolder_name
            target_dir.mkdir(exist_ok=True)

            # Copy files from test directory
            for file in subfolder.iterdir():
                if file.is_file():
                    target_file = target_dir / file.name
                    if not target_file.exists():
                        shutil.copy2(file, target_file)

            # Copy files from train directory with unique names if needed
            for file in train_subfolder.iterdir():
                if file.is_file():
                    target_file = target_dir / file.name
                    if target_file.exists():
                        # Add suffix to filename if it already exists
                        stem = file.stem
                        suffix = file.suffix
                        counter = 1
                        while target_file.exists():
                            new_name = f"{stem}_train_{counter}{suffix}"
                            target_file = target_dir / new_name
                            counter += 1

                    shutil.copy2(file, target_file)

            print(f"✅ Merged: {subfolder_name}")

    print(f"🎉 Merging completed! Result in: {merged_dir}")

    print("🗑️ Removing original test_crops and train_crops folders...")
    try:
        shutil.rmtree(test_crops_dir)
        print("✅ Removed test_crops")
        shutil.rmtree(train_crops_dir)
        print("✅ Removed train_crops")
    except Exception as e:
        print(f"⚠️ Error removing folders: {e}")


def cleanup_warp_directory():
    """Clean WaRP directory, keeping only merged_crops"""
    warp_dir = RAW_DIR / "WaRP"

    if not warp_dir.exists():
        print("WaRP folder not found for cleanup!")
        return

    print("🧹 Cleaning WaRP directory...")

    merged_crops_path = None

    # Find merged_crops in the directory structure
    for root, dirs, files in os.walk(warp_dir):
        if "merged_crops" in dirs:
            merged_crops_path = Path(root) / "merged_crops"
            break

    if not merged_crops_path or not merged_crops_path.exists():
        print("merged_crops folder not found!")
        return

    # Create temporary directory to save merged_crops
    temp_dir = warp_dir.parent / "WaRP_temp"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Move merged_crops to temporary directory
        temp_merged = temp_dir / "merged_crops"
        shutil.move(str(merged_crops_path), str(temp_merged))
        print("✅ merged_crops moved to temporary folder")

        shutil.rmtree(warp_dir)
        print("✅ WaRP folder removed")

        warp_dir.mkdir(parents=True, exist_ok=True)

        shutil.move(str(temp_merged), str(warp_dir / "merged_crops"))
        print("✅ merged_crops returned to cleaned WaRP folder")

        shutil.rmtree(temp_dir)
        print("✅ Temporary folder removed")

        print("🎉 WaRP folder successfully cleaned! Only merged_crops remains")

    except Exception as e:
        print(f"❌ Error cleaning WaRP folder: {e}")
        if temp_dir.exists():
            if (temp_dir / "merged_crops").exists():
                shutil.move(str(temp_dir / "merged_crops"),
                            str(warp_dir / "merged_crops"))
            shutil.rmtree(temp_dir)

def organize_dataset(dataset_name):
    """Universal function for organizing dataset structures"""
    dataset_dir = RAW_DIR / dataset_name
    
    if not dataset_dir.exists():
        print(f"Folder {dataset_name} not found!")
        return False
    
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        print(f"❌ Configuration for {dataset_name} not found!")
        return False
    
    print(f"\n📁 Organizing dataset: {dataset_name}")
    
    # Find source data folder
    source_dir = dataset_dir / config["source_path"]
    
    moved_count = 0
    classes_found = []
    
    for class_dir in source_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            
            # Check if this is a known class (case insensitive)
            class_name_lower = class_name.lower()
            config_classes_lower = [c.lower() for c in config["classes"]]
            
            if class_name_lower not in config_classes_lower:
                print(f"⚠️ Unknown class: {class_name}, skipping")
                continue
            
            # Find correct class name spelling from configuration
            correct_class_name = next((c for c in config["classes"] if c.lower() == class_name_lower), class_name)
            
            target_dir = dataset_dir / correct_class_name
            classes_found.append(correct_class_name)
            
            print(f"➡️ Processing: {class_name} -> {correct_class_name}")
            
            # If target folder already exists, merge contents
            if target_dir.exists():
                files_copied = 0
                for file_path in class_dir.glob("*.*"):
                    if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        target_file = target_dir / file_path.name
                        # If file already exists, add suffix
                        if target_file.exists():
                            stem = file_path.stem
                            suffix = file_path.suffix
                            counter = 1
                            while target_file.exists():
                                new_name = f"{stem}_{counter}{suffix}"
                                target_file = target_dir / new_name
                                counter += 1
                        shutil.copy2(file_path, target_file)
                        files_copied += 1
                        moved_count += 1
                print(f"  ✅ Merged {files_copied} files into {correct_class_name}")
            else:
                # Simply move the entire folder
                shutil.move(str(class_dir), str(target_dir))
                file_count = len([f for f in target_dir.glob("*.*") if f.is_file()])
                moved_count += file_count
                print(f"  ✅ Moved {file_count} files to {correct_class_name}")
    
    try:
        current_dir = source_dir
        while current_dir != dataset_dir:
            if current_dir.exists() and not any(current_dir.iterdir()):
                shutil.rmtree(current_dir)
                print(f"🗑️ Removed empty folder: {current_dir.name}")
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root
                break
            current_dir = parent_dir
    except Exception as e:
        print(f"⚠️ Failed to remove source folders: {e}")
    
    files_removed = 0
    for pattern in config["remove_files"]:
        for file_path in dataset_dir.glob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    files_removed += 1
                    print(f"🗑️ Removed file: {file_path.name}")
            except Exception as e:
                print(f"⚠️ Failed to remove {file_path.name}: {e}")
    
    if files_removed > 0:
        print(f"🗑️ Removed {files_removed} system files")
    
    print(f"\n🎉 Organization of {dataset_name} completed!")
    return True

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    for name, slug in DATASETS.items():
        out_dir = RAW_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} -> {out_dir}")
        api.dataset_download_files(slug, path=str(
            out_dir), unzip=True, quiet=False)
        print(f"✅ Done: {name}")

    if "WaRP" in DATASETS:
        merge_warp_c_folders()
        cleanup_warp_directory()

    datasets_to_organize = ["garbage_classification_1", "garbage_classification_2", "realwaste"]
    for dataset_name in datasets_to_organize:
        if dataset_name in DATASETS:
            organize_dataset(dataset_name)


if __name__ == "__main__":
    main()