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
            'battery', 'clothes', 'lamp', 'biological'
        }

        self.class_mapping = self._build_class_mapping()
        
        # Конфигурация для каждого датасета
        self.dataset_configs = self._build_dataset_configs()

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
            'shoes': 'clothes',
            'biological': 'biological'
        }

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
            'Vegetation': 'biological'  # Исправлена опечатка
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
        """Создает конфигурацию для каждого датасета"""
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

    def _process_dataset(self, dataset_key):
        """Универсальная функция для обработки любого датасета"""
        config = self.dataset_configs.get(dataset_key)
        if not config:
            print(f"❌ Конфигурация для {dataset_key} не найдена!")
            return pd.DataFrame()

        dataset_path = self.raw_dir / config['path']
        records = []

        if not dataset_path.exists():
            print(f"❌ Путь не найден: {dataset_path}")
            return pd.DataFrame()

        print(f"📁 Обрабатываем {dataset_key} из: {dataset_path}")

        total_images = 0
        classes_processed = 0

        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                original_class = class_dir.name
                
                # Для некоторых датасетов приводим к нижнему регистру
                if dataset_key in ['12classes', 'garbage_classification_1', 'garbage_classification_2', 'trash_type', 'garbage_dataset']:
                    original_class = original_class.lower()
                
                unified_class = self.class_mapping[config['prefix']].get(original_class)

                if not unified_class:
                    print(f"⚠️ Неизвестный класс в {dataset_key}: {original_class}")
                    continue

                print(f"  📂 Обрабатываем класс: {original_class} -> {unified_class}")

                image_count = 0
                for pattern in config['file_patterns']:
                    for img_path in class_dir.glob(pattern):
                        metrics = self._calculate_image_metrics(img_path)
                        if not metrics:
                            continue

                        # Создаем уникальный ID изображения
                        image_id = f"{config['prefix']}_{original_class.replace(' ', '_')}_{img_path.stem}"
                        
                        records.append({
                            'image_id': image_id,
                            'file_path': str(img_path.relative_to(self.raw_dir)),
                            'dataset': config['prefix'],
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
                        total_images += 1

                print(f"    ✅ {original_class}: {image_count} изображений")
                classes_processed += 1

        print(f"🎉 Обработка {dataset_key} завершена: {total_images} изображений в {classes_processed} классах")
        return pd.DataFrame(records)
    
    def assign_splits(self, df, train_size=0.6, val_size=0.2, test_size=0.2):
        """Назначает train/val/test splits с стратификацией по классам"""
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
        """Основной метод объединения датасетов"""
        print("Starting dataset unification...")

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