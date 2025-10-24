import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from image_quality_analyzer import ImageQualityAnalyzer


class DatasetUnifier:
    def __init__(self, raw_dir="data/raw", unified_dir="data/unified"):
        self.raw_dir = Path(raw_dir)
        self.unified_dir = Path(unified_dir)
        self.manifest_path = self.unified_dir / "manifest.csv"
        self.quality_analyzer = ImageQualityAnalyzer()

        self.unified_classes = {
            'cardboard', 'paper', 'plastic', 'metal', 'glass', 'trash',
            'battery', 'clothes', 'lamp', 'biological'
        }

        self.class_mapping = self._build_class_mapping()

        self.dataset_configs = self._build_dataset_configs()

    def _build_class_mapping(self):
        """Build class mapping from different datasets to unified classes"""
        mapping = {}

        mapping['trashnet'] = {
            'cardboard': 'cardboard',
            'paper': 'paper',
            'plastic': 'plastic',
            'metal': 'metal',
            'glass': 'glass',
            'trash': 'trash'
        }

        mapping['12classes'] = {
            'cardboard': 'cardboard',
            'paper': 'paper',
            'plastic': 'plastic',
            'metal': 'metal',
            'trash': 'trash',
            'brown-glass': 'glass',
            'green-glass': 'glass',
            'white-glass': 'glass',
            'battery': 'battery',
            'clothes': 'clothes',
            'shoes': 'clothes',
            'biological': 'biological'
        }

        mapping['WaRP'] = {
            # Plastic - all bottle and canister
            'bottle-blue': 'plastic',
            'bottle-blue-full': 'plastic',
            'bottle-blue5l': 'plastic',
            'bottle-blue5l-full': 'plastic',
            'bottle-dark': 'plastic',
            'bottle-dark-full': 'plastic',
            'bottle-green': 'plastic',
            'bottle-green-full': 'plastic',
            'bottle-milk': 'plastic',
            'bottle-milk-full': 'plastic',
            'bottle-multicolor': 'plastic',
            'bottle-multicolor-full': 'plastic',
            'bottle-oil': 'plastic',
            'bottle-oil-full': 'plastic',
            'bottle-transp': 'plastic',
            'bottle-transp-full': 'plastic',
            'bottle-yogurt': 'plastic',
            'canister': 'plastic',

            # Glass
            'glass-dark': 'glass',
            'glass-green': 'glass',
            'glass-transp': 'glass',

            # Metal
            'cans': 'metal',

            # Cardboard
            'juice-cardboard': 'cardboard',
            'milk-cardboard': 'cardboard',

            # Detergents (plastic)
            'detergent': 'plastic',
            'detergent-box': 'plastic',
            'detergent-color': 'plastic',
            'detergent-transparent': 'plastic',
            'detergent-white': 'plastic'
        }

        mapping['garbage_classification_1'] = {
            'cardboard': 'cardboard',
            'glass': 'glass',
            'metal': 'metal',
            'paper': 'paper',
            'plastic': 'plastic',
            'trash': 'trash'
        }

        mapping['trash_type'] = {
            'cardboard': 'cardboard',
            'glass': 'glass',
            'metal': 'metal',
            'paper': 'paper',
            'plastic': 'plastic',
            'trash': 'trash'
        }

        mapping['realwaste'] = {
            'Cardboard': 'cardboard',
            'Food Organics': 'biological',
            'Glass': 'glass',
            'Metal': 'metal',
            'Miscellaneous Trash': 'trash',
            'Paper': 'paper',
            'Plastic': 'plastic',
            'Textile Trash': 'clothes',
            'Vegetation': 'biological'
        }

        mapping['garbage_classification_2'] = {
            'cardboard': 'cardboard',
            'glass': 'glass',
            'metal': 'metal',
            'paper': 'paper',
            'plastic': 'plastic',
            'trash': 'trash'
        }

        mapping['garbage_dataset'] = {
            'battery': 'battery',
            'biological': 'biological',
            'cardboard': 'cardboard',
            'clothes': 'clothes',
            'glass': 'glass',
            'metal': 'metal',
            'paper': 'paper',
            'plastic': 'plastic',
            'shoes': 'clothes',
            'trash': 'trash'
        }

        return mapping

    def _build_dataset_configs(self):
        """Create configuration for each dataset"""
        return {
            'trashnet': {
                'path': 'trashnet/dataset-resized',
                'prefix': 'trashnet',
                'file_patterns': ['*.jpg']
            },
            '12classes': {
                'path': '12classes/garbage_classification',
                'prefix': '12classes',
                'file_patterns': ['*.jpg', '*.jpeg', '*.png']
            },
            'WaRP': {
                'path': 'WaRP/merged_crops',
                'prefix': 'WaRP',
                'file_patterns': ['*.jpg', '*.jpeg', '*.png']
            },
            'garbage_classification_1': {
                'path': 'garbage_classification_1',
                'prefix': 'garbage_classification_1',
                'file_patterns': ['*.jpg', '*.jpeg', '*.png']
            },
            'trash_type': {
                'path': 'trash_type/TrashType_Image_Dataset',
                'prefix': 'trash_type',
                'file_patterns': ['*.jpg', '*.jpeg', '*.png']
            },
            'realwaste': {
                'path': 'realwaste',
                'prefix': 'realwaste',
                'file_patterns': ['*.jpg', '*.jpeg', '*.png']
            },
            'garbage_classification_2': {
                'path': 'garbage_classification_2',
                'prefix': 'garbage_classification_2',
                'file_patterns': ['*.jpg', '*.jpeg', '*.png']
            },
            'garbage_dataset': {
                'path': 'garbage_dataset/garbage-dataset',
                'prefix': 'garbage_dataset',
                'file_patterns': ['*.jpg', '*.jpeg', '*.png']
            }
        }

    def _process_dataset(self, dataset_key):
        """Universal function for processing any dataset"""
        config = self.dataset_configs.get(dataset_key)
        if not config:
            print(f"❌ Configuration for {dataset_key} not found!")
            return pd.DataFrame()

        dataset_path = self.raw_dir / config['path']
        records = []

        if not dataset_path.exists():
            print(f"❌ Path not found: {dataset_path}")
            return pd.DataFrame()

        print(f"📁 Processing {dataset_key} from: {dataset_path}")

        total_images = 0
        classes_processed = 0

        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                original_class = class_dir.name

                if dataset_key in ['12classes', 'garbage_classification_1', 'garbage_classification_2', 'trash_type', 'garbage_dataset']:
                    original_class = original_class.lower()

                unified_class = self.class_mapping[config['prefix']].get(
                    original_class)

                if not unified_class:
                    print(
                        f"⚠️ Unknown class in {dataset_key}: {original_class}")
                    continue

                print(
                    f"  📂 Processing class: {original_class} -> {unified_class}")

                image_count = 0
                for pattern in config['file_patterns']:
                    for img_path in class_dir.glob(pattern):
                        metrics = self.quality_analyzer.analyze_image(img_path)
                        if not metrics:
                            continue

                        # Create unique image ID
                        image_id = f"{config['prefix']}_{original_class.replace(' ', '_')}_{img_path.stem}"

                        records.append({
                            # Basic information (5)
                            'image_id': image_id,
                            'file_path': str(img_path.relative_to(self.raw_dir)),
                            'dataset': config['prefix'],
                            'unified_class': unified_class,
                            'split': '',

                            # Basic size metrics (5)
                            'width': metrics['width'],
                            'height': metrics['height'],
                            'aspect_ratio': metrics['aspect_ratio'],
                            'total_pixels': metrics['total_pixels'],
                            'channels': metrics['channels'],

                            # Main quality metrics (6)
                            'quality_score': metrics['quality_score'],
                            'brightness_score': metrics['brightness_score'],
                            'contrast_score': metrics['contrast_score'],
                            'edge_score': metrics['edge_score'],
                            'noise_score': metrics['noise_score'],
                            'blur_score': metrics['blur_score'],

                            # Color metrics (4)
                            'saturation': metrics['saturation'],
                            'color_balance_bias': metrics['color_balance_bias'],
                            'is_color_cast': metrics['is_color_cast'],
                            'is_monochrome': metrics['is_monochrome'],

                            # Exposure metrics (3)
                            'overexposed_ratio': metrics['overexposed_ratio'],
                            'underexposed_ratio': metrics['underexposed_ratio'],
                            'dynamic_range': metrics['dynamic_range'],

                            # Problem detection for RL (10)
                            'needs_contrast_boost': metrics['needs_contrast_boost'],
                            'needs_brightness_fix': metrics['needs_brightness_fix'],
                            'needs_sharpening': metrics['needs_sharpening'],
                            'needs_denoising': metrics['needs_denoising'],
                            'needs_deblurring': metrics['needs_deblurring'],
                            'needs_saturation_boost': metrics['needs_saturation_boost'],
                            'needs_color_balance': metrics['needs_color_balance'],
                            'needs_exposure_fix': metrics['needs_exposure_fix'],
                            'is_low_entropy': metrics['is_low_entropy'],
                            'has_color_cast': metrics['has_color_cast'],

                            # RL feature (1)
                            'quality_level': metrics['quality_level']
                        })
                        image_count += 1
                        total_images += 1

                print(f"    ✅ {original_class}: {image_count} images")
                classes_processed += 1

        print(
            f"🎉 Processing {dataset_key} completed: {total_images} images in {classes_processed} classes")
        return pd.DataFrame(records)

    def assign_splits(self, df, train_size=0.6, val_size=0.2, test_size=0.2):
        """Assign train/val/test splits with class stratification"""
        if len(df) == 0:
            return df

        train_val_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            stratify=df['unified_class'],
            random_state=42
        )

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size/(1-test_size),
            stratify=df.loc[train_val_idx, 'unified_class'],
            random_state=42
        )

        df.loc[train_idx, 'split'] = 'train'
        df.loc[val_idx, 'split'] = 'val'
        df.loc[test_idx, 'split'] = 'test'

        return df

    def unify_datasets(self):
        """Main method for dataset unification"""
        print("Starting dataset unification with advanced quality analysis...")

        all_data = pd.DataFrame()

        datasets_to_process = [
            'trashnet',
            '12classes',
            'WaRP',
            'garbage_classification_1',
            'trash_type',
            'realwaste',
            'garbage_classification_2',
            'garbage_dataset'
        ]

        for dataset_key in datasets_to_process:
            print(f"\n{'='*50}")
            df = self._process_dataset(dataset_key)
            all_data = pd.concat([all_data, df], ignore_index=True)
            print(f"✅ {dataset_key}: {len(df)} images")
            print(f"{'='*50}")

        print(f"\nTotal records collected: {len(all_data)}")

        if len(all_data) == 0:
            print("No records found! Check if datasets are downloaded correctly.")
            return 0

        print("Assigning train/val/test splits...")
        all_data = self.assign_splits(all_data)

        print(f"Saving manifest to {self.manifest_path}...")
        self.unified_dir.mkdir(parents=True, exist_ok=True)

        all_data.to_csv(self.manifest_path, index=False, encoding='utf-8')

        # Extended statistics
        print("\n=== Dataset Statistics ===")
        print(f"Total images: {len(all_data)}")
        print(f"By dataset:\n{all_data['dataset'].value_counts()}")
        print(f"By class:\n{all_data['unified_class'].value_counts()}")
        print(f"By split:\n{all_data['split'].value_counts()}")

        # Image quality
        print(f"\n=== Image Quality Statistics ===")
        print(f"Average quality score: {all_data['quality_score'].mean():.3f}")
        print(f"Average brightness: {all_data['brightness_score'].mean():.3f}")
        print(f"Average contrast: {all_data['contrast_score'].mean():.3f}")
        print(f"Average saturation: {all_data['saturation'].mean():.3f}")
        print(f"Average noise: {all_data['noise_score'].mean():.3f}")
        print(f"Average blur: {all_data['blur_score'].mean():.3f}")

        # RL statistics
        print(f"\n=== RL-specific Statistics ===")
        print(
            f"Images needing processing: {all_data['needs_any_processing'].sum()} ({all_data['needs_any_processing'].mean()*100:.1f}%)")
        print(f"Quality level distribution:")
        quality_dist = all_data['quality_level'].value_counts().sort_index()
        for level, count in quality_dist.items():
            level_names = {0: 'Low', 1: 'Medium', 2: 'High'}
            print(
                f"  - {level_names[level]}: {count} images ({count/len(all_data)*100:.1f}%)")

        # Problem statistics
        print(f"\n=== Common Problems ===")
        problem_cols = [col for col in all_data.columns if col.startswith(
            'needs_') or col.startswith('has_')]
        problem_stats = []
        for col in problem_cols:
            count = all_data[col].sum()
            if count > 0:
                percentage = count/len(all_data)*100
                problem_stats.append((col, count, percentage))

        return len(all_data)


def main():
    unifier = DatasetUnifier()
    total_images = unifier.unify_datasets()
    print(f"\nUnification complete! Processed {total_images} images.")


if __name__ == "__main__":
    main()
