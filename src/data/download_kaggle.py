import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil

DATASETS = {
    "trashnet": "feyzazkefe/trashnet",
    "WaRP": "parohod/warp-waste-recycling-plant-dataset",
    "12classes": "mostafaabla/garbage-classification"
}
RAW_DIR = Path("data/raw")

def merge_warp_c_folders():
    """Объединяет папки test_crops и train_crops в Warp-C"""
    warp_dir = RAW_DIR / "WaRP"
    
    if not warp_dir.exists():
        print("Папка WaRP не найдена!")
        return
    
    warp_c_dir = warp_dir / "Warp-C"
    if not warp_c_dir.exists():
        print("Папка Warp-C не найдена!")
        return
    
    test_crops_dir = warp_c_dir / "test_crops"
    train_crops_dir = warp_c_dir / "train_crops"
    
    if not test_crops_dir.exists() or not train_crops_dir.exists():
        print("Не найдены папки test_crops или train_crops!")
        return
    
    merged_dir = warp_c_dir / "merged_crops"
    merged_dir.mkdir(exist_ok=True)
    
    for category in test_crops_dir.iterdir():
        if not category.is_dir():
            continue
            
        print(f"📁 Обрабатываем категорию: {category.name}")
        
        train_category_dir = train_crops_dir / category.name
        
        if not train_category_dir.exists():
            print(f"⚠️ Категория {category.name} не найдена в train_crops, пропускаем")
            continue
        
        for subfolder in category.iterdir():
            if not subfolder.is_dir():
                continue
                
            subfolder_name = subfolder.name
            train_subfolder = train_category_dir / subfolder_name
            
            if not train_subfolder.exists():
                print(f"⚠️ Подпапка {subfolder_name} не найдена в train, пропускаем")
                continue
            
            target_dir = merged_dir / subfolder_name
            target_dir.mkdir(exist_ok=True)
            
            for file in subfolder.iterdir():
                if file.is_file():
                    target_file = target_dir / file.name
                    if not target_file.exists():
                        shutil.copy2(file, target_file)
            
            for file in train_subfolder.iterdir():
                if file.is_file():
                    # Генерируем уникальное имя, если файл уже существует
                    target_file = target_dir / file.name
                    if target_file.exists():
                        # Добавляем суффикс к имени файла
                        stem = file.stem
                        suffix = file.suffix
                        counter = 1
                        while target_file.exists():
                            new_name = f"{stem}_train_{counter}{suffix}"
                            target_file = target_dir / new_name
                            counter += 1
                    
                    shutil.copy2(file, target_file)
            
            print(f"✅ Объединено: {subfolder_name}")
    
    print(f"🎉 Объединение завершено! Результат в: {merged_dir}")
    
    print("🗑️ Удаляем исходные папки test_crops и train_crops...")
    try:
        shutil.rmtree(test_crops_dir)
        print("✅ Удалена test_crops")
        shutil.rmtree(train_crops_dir)
        print("✅ Удалена train_crops")
    except Exception as e:
        print(f"⚠️ Ошибка при удалении папок: {e}")

def cleanup_warp_directory():
    """Очищает папку WaRP, оставляя только merged_crops"""
    warp_dir = RAW_DIR / "WaRP"
    
    if not warp_dir.exists():
        print("Папка WaRP не найдена для очистки!")
        return
    
    print("🧹 Очищаем папку WaRP...")
    
    # Находим папку merged_crops (может быть в разных местах)
    merged_crops_path = None
    
    # Ищем merged_crops в структуре
    for root, dirs, files in os.walk(warp_dir):
        if "merged_crops" in dirs:
            merged_crops_path = Path(root) / "merged_crops"
            break
    
    if not merged_crops_path or not merged_crops_path.exists():
        print("Папка merged_crops не найдена!")
        return
    
    # Создаем временную папку для сохранения merged_crops
    temp_dir = warp_dir.parent / "WaRP_temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Перемещаем merged_crops во временную папку
        temp_merged = temp_dir / "merged_crops"
        shutil.move(str(merged_crops_path), str(temp_merged))
        print("✅ merged_crops перемещена во временную папку")
        
        shutil.rmtree(warp_dir)
        print("✅ Папка WaRP удалена")
        
        warp_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(temp_merged), str(warp_dir / "merged_crops"))
        print("✅ merged_crops возвращена в очищенную папку WaRP")
        
        shutil.rmtree(temp_dir)
        print("✅ Временная папка удалена")
        
        print("🎉 Папка WaRP успешно очищена! Осталась только merged_crops")
        
    except Exception as e:
        print(f"❌ Ошибка при очистке папки WaRP: {e}")
        if temp_dir.exists():
            if (temp_dir / "merged_crops").exists():
                shutil.move(str(temp_dir / "merged_crops"), str(warp_dir / "merged_crops"))
            shutil.rmtree(temp_dir)

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    for name, slug in DATASETS.items():
        out_dir = RAW_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} -> {out_dir}")
        api.dataset_download_files(slug, path=str(out_dir), unzip=True, quiet=False)
        print(f"✅ Done: {name}")
    
    if "WaRP" in DATASETS:
        merge_warp_c_folders()
        cleanup_warp_directory()

if __name__ == "__main__":
    main()