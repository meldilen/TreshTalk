import os
import csv
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class DatasetUnifier:
    def __init__(self, raw_dir="data/raw", unified_dir="data/unified"):
        self.raw_dir = Path(raw_dir)
        self.unified_dir = Path(unified_dir)
        self.manifest_path = self.unified_dir / "manifest.csv"

        self.unified_classes = {
            'cardboard', 'paper', 'plastic', 'metal', 'glass', 'trash',
            'battery', 'clothes', 'shoes', 'lamp', 'biological'
        }

        self.class_mapping = self._build_class_mapping()

    def _build_class_mapping(self):
        """Создает маппинг классов из разных датасетов к унифицированным"""
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
            'shoes': 'shoes',
            'biological': 'biological'
        }

        # WaRP mapping (упрощенный)
        mapping['WaRP'] = {
            # Пластик - все bottle и canister
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

            # Стекло
            'glass-dark': 'glass',
            'glass-green': 'glass',
            'glass-transp': 'glass',

            # Металл
            'cans': 'metal',

            # Картон
            'juice-cardboard': 'cardboard',
            'milk-cardboard': 'cardboard',

            # Моющие средства (пластик)
            'detergent': 'plastic',
            'detergent-box': 'plastic',
            'detergent-color': 'plastic',
            'detergent-transparent': 'plastic',
            'detergent-white': 'plastic'
        }

        return mapping

    def _calculate_image_metrics(self, image_path):
        """Вычисляет метрики качества изображения"""
        try:
            with Image.open(image_path) as img:
                img_array = np.array(img)

                # Базовые метрики
                width, height = img.size
                format_type = img.format if img.format else 'UNKNOWN'

                # Яркость (среднее значение пикселей в grayscale)
                if len(img_array.shape) == 3:
                    gray = np.mean(img_array, axis=2)
                else:
                    gray = img_array
                brightness = np.mean(gray) / 255.0

                # Контраст (стандартное отклонение)
                contrast = np.std(gray) / 255.0

                # Edge score (простой детектор границ)
                dy, dx = np.gradient(gray.astype(float))
                edge_score = np.mean(np.sqrt(dx**2 + dy**2)) / 255.0

                # Noise score (вариация в маленьких участках)
                noise_score = self._estimate_noise(gray)

                # Общий quality score
                quality_score = (
                    0.3 * (1 - abs(brightness - 0.5)) +  # Яркость близка к 0.5
                    0.3 * min(contrast * 3, 1.0) +       # Хороший контраст
                    0.2 * min(edge_score * 5, 1.0) +     # Четкие границы
                    0.2 * (1 - min(noise_score * 10, 1.0))  # Низкий шум
                )

                return {
                    'width': width,
                    'height': height,
                    'format': format_type,
                    'quality_score': round(quality_score, 4),
                    'brightness_score': round(brightness, 4),
                    'contrast_score': round(contrast, 4),
                    'edge_score': round(edge_score, 4),
                    'noise_score': round(noise_score, 4)
                }
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def _estimate_noise(self, gray_image):
        """Оценивает уровень шума в изображении"""
        try:
            # Простой метод оценки шума через вариацию в маленьких блоках
            h, w = gray_image.shape
            block_size = 8
            variances = []

            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray_image[i:i+block_size, j:j+block_size]
                    variances.append(np.var(block))

            return np.mean(variances) / 255.0 if variances else 0.0
        except:
            return 0.0

    def _process_trashnet(self):
        """Обрабатывает TrashNet датасет"""
        trashnet_path = self.raw_dir / "trashnet" / "dataset-resized"
        records = []

        if not trashnet_path.exists():
            print(f"TrashNet path not found: {trashnet_path}")
            return pd.DataFrame(records)

        for class_dir in trashnet_path.iterdir():
            if class_dir.is_dir():
                original_class = class_dir.name
                unified_class = self.class_mapping['trashnet'].get(
                    original_class)

                if not unified_class:
                    continue

                for img_path in class_dir.glob("*.jpg"):
                    metrics = self._calculate_image_metrics(img_path)
                    if not metrics:
                        continue

                    records.append({
                        'image_id': f"trashnet_{img_path.stem}",
                        'file_path': str(img_path.relative_to(self.raw_dir)),
                        'dataset': 'trashnet',
                        'format': metrics['format'],
                        'unified_class': unified_class,
                        'width': metrics['width'],
                        'height': metrics['height'],
                        'split': '',  # Будет заполнено позже
                        'quality_score': metrics['quality_score'],
                        'brightness_score': metrics['brightness_score'],
                        'contrast_score': metrics['contrast_score'],
                        'edge_score': metrics['edge_score'],
                        'noise_score': metrics['noise_score']
                    })

        return pd.DataFrame(records)

    def _process_garbage12(self):
        """Обрабатывает Garbage Classification (12 classes) датасет"""
        garbage12_path = self.raw_dir / "12classes" / "garbage_classification"
        records = []

        if not garbage12_path.exists():
            print(f"Garbage12 path not found: {garbage12_path}")
            return pd.DataFrame(records)

        for class_dir in garbage12_path.iterdir():
            if class_dir.is_dir():
                original_class = class_dir.name.lower()
                unified_class = self.class_mapping['12classes'].get(
                    original_class)

                if not unified_class:
                    print(f"Unknown class in garbage12: {original_class}")
                    continue

                for img_path in class_dir.glob("*.*"):
                    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                        continue

                    metrics = self._calculate_image_metrics(img_path)
                    if not metrics:
                        continue

                    records.append({
                        'image_id': f"garbage12_{img_path.stem}",
                        'file_path': str(img_path.relative_to(self.raw_dir)),
                        'dataset': '12classes',
                        'format': metrics['format'],
                        'unified_class': unified_class,
                        'width': metrics['width'],
                        'height': metrics['height'],
                        'split': '',
                        'quality_score': metrics['quality_score'],
                        'brightness_score': metrics['brightness_score'],
                        'contrast_score': metrics['contrast_score'],
                        'edge_score': metrics['edge_score'],
                        'noise_score': metrics['noise_score']
                    })

        return pd.DataFrame(records)

    def _process_warp(self):
        """Обрабатывает WaRP датасет из merged_crops"""
        warp_path = self.raw_dir / "WaRP" / "merged_crops"
        records = []

        if not warp_path.exists():
            print(f"WaRP merged_crops path not found: {warp_path}")
            return pd.DataFrame(records)

        # Обрабатываем каждую папку с классами в merged_crops
        for class_dir in warp_path.iterdir():
            if class_dir.is_dir():
                original_class = class_dir.name
                unified_class = self.class_mapping['WaRP'].get(original_class)

                if not unified_class:
                    print(f"Unknown class in WaRP merged_crops: {original_class}")
                    continue

                print(
                    f"Processing WaRP class: {original_class} -> {unified_class}")

                image_count = 0
                # Обрабатываем все изображения в папке класса
                for img_path in class_dir.glob("*.*"):
                    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                        continue

                    metrics = self._calculate_image_metrics(img_path)
                    if not metrics:
                        continue

                    records.append({
                        'image_id': f"warp_{original_class}_{img_path.stem}",
                        'file_path': str(img_path.relative_to(self.raw_dir)),
                        'dataset': 'WaRP',
                        'format': metrics['format'],
                        'unified_class': unified_class,
                        'width': metrics['width'],
                        'height': metrics['height'],
                        'split': '',
                        'quality_score': metrics['quality_score'],
                        'brightness_score': metrics['brightness_score'],
                        'contrast_score': metrics['contrast_score'],
                        'edge_score': metrics['edge_score'],
                        'noise_score': metrics['noise_score']
                    })
                    image_count += 1

                print(f"  - {original_class}: {image_count} images")

        print(f"Processed {len(records)} total images from WaRP merged_crops")
        return pd.DataFrame(records)

    def assign_splits(self, df, test_size=0.2, val_size=0.1):
        """Назначает train/val/test splits с стратификацией по классам"""
        if len(df) == 0:
            return df

        # Сначала разделяем на train+val и test
        train_val_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            stratify=df['unified_class'],
            random_state=42
        )

        # Затем разделяем train+val на train и val
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
        """Основной метод объединения датасетов"""
        print("Starting dataset unification...")

        all_data = pd.DataFrame()

        print("Processing TrashNet...")
        trashnet_df = self._process_trashnet()
        all_data = pd.concat([all_data, trashnet_df], ignore_index=True)

        print("Processing Garbage12...")
        garbage12_df = self._process_garbage12()
        all_data = pd.concat([all_data, garbage12_df], ignore_index=True)

        print("Processing WaRP...")
        warp_df = self._process_warp()
        all_data = pd.concat([all_data, warp_df], ignore_index=True)

        print(f"Total records collected: {len(all_data)}")

        if len(all_data) == 0:
            print("No records found! Check if datasets are downloaded correctly.")
            return 0

        print("Assigning train/val/test splits...")
        all_data = self.assign_splits(all_data)

        print(f"Saving manifest to {self.manifest_path}...")
        self.unified_dir.mkdir(parents=True, exist_ok=True)
        
        all_data.to_csv(self.manifest_path, index=False, encoding='utf-8')

        # Статистика
        print("\n=== Dataset Statistics ===")
        print(f"Total images: {len(all_data)}")
        print(f"By dataset:\n{all_data['dataset'].value_counts()}")
        print(f"By class:\n{all_data['unified_class'].value_counts()}")
        print(f"By split:\n{all_data['split'].value_counts()}")

        # Качество изображений
        print(f"\nImage quality stats:")
        print(f"Average quality score: {all_data['quality_score'].mean():.3f}")
        print(f"Average brightness: {all_data['brightness_score'].mean():.3f}")
        print(f"Average contrast: {all_data['contrast_score'].mean():.3f}")

        return len(all_data)


def main():
    unifier = DatasetUnifier()
    total_images = unifier.unify_datasets()
    print(f"\nUnification complete! Processed {total_images} images.")


if __name__ == "__main__":
    main()
