import gym
from gym import spaces
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import os
import tempfile
from typing import Tuple, Dict, Any
import cv2

# Отложенный импорт внутри функции
def evaluate_single_image(*args, **kwargs):
    from src.rl.eval import evaluate_single_image as eval_func
    return eval_func(*args, **kwargs)

class WasteRLMultiStepEnv(gym.Env):
    """
    Gym-среда для пошагового адаптивного препроцессинга изображения.
    Каждый эпизод:
      - выбирается случайная запись из manifest (строка с полями file_path/unified_class и пр.)
      - агент имеет max_steps попыток применить дискретные действия к изображению
      - после каждого действия вычисляется reward на основе классификатора (например, уверенность/корректность)
    Observation:
      Непрерывный вектор с признаками изображения (яркость, контраст, шум и т.п.)
    Action:
      Дискретный набор действий (brighten, darken, sharpen, denoise, crop, stop)
    Возвращаемые значения:
      reset(...) -> observation, info
      step(action) -> observation, reward, terminated, truncated, info
    """

    metadata = {"render.modes": []}

    def __init__(self,
                 manifest_df,
                 model,
                 device,
                 class2idx,
                 data_root: str,
                 max_steps: int = 5):
        super().__init__()

        self.df = manifest_df.reset_index(drop=True)
        self.model = model
        self.device = device
        self.class2idx = class2idx
        self.data_root = data_root
        self.max_steps = max_steps

        # Список признаков, которые будут в observation (все в [0,1] после нормировки ниже)
        self.state_features = [
            "brightness", "contrast", "edge_strength",
            "entropy", "mean_color_saturation", "noise_level"
        ]

        # Дискретный набор действий. Включаем 'noop' и 'stop'.
        self.actions = [
            "noop", "brighten", "darken", "contrast_boost",
            "sharpen", "denoise_median", "saturation_boost",
            "crop_center", "exposure_fix", "stop"
        ]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(self.state_features),), dtype=np.float32)

        # Переменные эпизода
        self.current_row = None
        self.original_image_path = None
        self.current_image_path = None
        self.true_label = None
        self.current_step = 0
        self.terminated = False
        self.truncated = False

    # Современный интерфейс gym.reset
    def reset(self, *, seed: int = None, options: dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_step = 0
        self.terminated = False
        self.truncated = False

        row = self.df.sample(1).iloc[0]
        self.current_row = row
        self.true_label = row["unified_class"]
        self.original_image_path = os.path.join(self.data_root, row["file_path"])
        # стартовое изображение — оригинал
        self.current_image_path = self.original_image_path

        # Состояние — вычисляемые признаки текущего изображения
        obs = self._compute_image_features(self.current_image_path)
        info = {"file_path": row["file_path"], "true_label": self.true_label}
        return obs.astype(np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Выполняем действие:
          - применяем действие к текущему изображению (сохраняем во временный файл)
          - оцениваем модель до и после (evaluate_single_image) -> формируем reward
          - обновляем текущее изображение и состояние
        Возвращаем observation, reward, terminated, truncated, info
        """
        assert self.current_image_path is not None, "Call reset() before step()"
        action_name = self.actions[int(action)]
        self.current_step += 1

        # Оценка до (accuracy 0/1 и confidence)
        old_correct, old_conf, _ = evaluate_single_image(self.model, self.device, self.current_image_path, self.true_label, self.class2idx)

        # Применяем действие (сохраняем в безопасный временный файл)
        new_path = self._apply_action_to_image(self.current_image_path, action_name)
        # обновляем текущий путь (следующее состояние основано на новом изображении)
        self.current_image_path = new_path

        # Оценка после
        new_correct, new_conf, _ = evaluate_single_image(self.model, self.device, self.current_image_path, self.true_label, self.class2idx)

        # Reward: комбинация бинарного улучшения (0/1) и разницы уверенности.
        # Эта формула даёт более плотную награду, чем только 0/1.
        reward = (new_correct - old_correct) + (new_conf - old_conf)
        # Малый штраф за шаг (чтобы поощрять короткие последовательности)
        reward -= 0.01

        # Если действие stop — завершаем эпизод (terminated=True)
        if action_name == "stop" or self.current_step >= self.max_steps:
            self.terminated = True

        # Ограничение по длине эпизода (truncated может быть True при time-limit)
        self.truncated = self.current_step >= self.max_steps

        # Новое наблюдение — вычисляем признаки для нового текущего изображения
        next_obs = self._compute_image_features(self.current_image_path)

        info = {
            "step": self.current_step,
            "action": action_name,
            "reward": reward,
            "old_acc": old_correct,
            "new_acc": new_correct,
            "old_conf": old_conf,
            "new_conf": new_conf,
            "file_path": self.current_row["file_path"]
        }

        return next_obs.astype(np.float32), float(reward), bool(self.terminated), bool(self.truncated), info

    def _apply_action_to_image(self, image_path: str, action: str) -> str:
        """
        Простые детерминированные трансформации. Сохраняем результат во временный файл.
        Возвращаем путь к новому изображению.
        """
        img = Image.open(image_path).convert("RGB")

        if action == "noop":
            out = img
        elif action == "brighten":
            out = ImageEnhance.Brightness(img).enhance(1.25)
        elif action == "darken":
            out = ImageEnhance.Brightness(img).enhance(0.8)
        elif action == "contrast_boost":
            out = ImageEnhance.Contrast(img).enhance(1.3)
        elif action == "sharpen":
            out = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150))
        elif action == "denoise_median":
            out = img.filter(ImageFilter.MedianFilter(size=3))
        elif action == "saturation_boost":
            out = ImageEnhance.Color(img).enhance(1.2)
        elif action == "crop_center":
            w, h = img.size
            cw, ch = int(w * 0.9), int(h * 0.9)
            left = (w - cw) // 2
            top = (h - ch) // 2
            out = img.crop((left, top, left + cw, top + ch)).resize((w, h))
        elif action == "exposure_fix":
            out = ImageEnhance.Brightness(img).enhance(1.05)
        elif action == "stop":
            out = img
        else:
            out = img

        # сохраняем во временный файл (удобно для параллельных эпизодов)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix="rl_img_")
        out.save(tmp.name, format="JPEG", quality=90)
        tmp.close()
        return tmp.name

    def _compute_image_features(self, image_path: str) -> np.ndarray:
        """
        Простая функция извлечения числовых признаков изображения.
        Возвращает вектор длины len(self.state_features) в диапазоне [0,1].
        Признаки (пример):
          - brightness: средняя яркость
          - contrast: стандартное отклонение яркости
          - edge_strength: средняя величина градиента (cv2.Sobel)
          - entropy: информационная энтропия гистограммы
          - mean_color_saturation: средняя насыщенность в HSV
          - noise_level: простой proxy — высокочастотная энергия (Laplacian)
        """
        img = cv2.imread(image_path)
        if img is None:
            # fallback — пустой вектор
            return np.zeros(len(self.state_features), dtype=np.float32)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # brightness and contrast proxy
        brightness = float(np.clip(np.mean(gray), 0.0, 1.0))
        contrast = float(np.clip(np.std(gray), 0.0, 1.0))

        # edge_strength: средняя абсолютная величина градиента (Sobel)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_strength = float(np.clip(np.mean(np.sqrt(gx*gx + gy*gy)), 0.0, 1.0))

        # entropy (гистограмма)
        hist = cv2.calcHist([gray], [0], None, [256], [0,1]).flatten()
        hist = hist / (hist.sum() + 1e-9)
        entropy = -float(np.sum([p * np.log2(p + 1e-9) for p in hist]))

        # mean_color_saturation
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = float(np.clip(np.mean(hsv[:,:,1]) / 255.0, 0.0, 1.0))

        # noise_level proxy: variance of Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        noise_level = float(np.clip(np.var(lap), 0.0, 1.0))

        vec = np.array([brightness, contrast, edge_strength, entropy / 8.0, saturation, noise_level], dtype=np.float32)
        # нормировка entropy (пример) — entropy/8 чтобы быть ~в диапазоне [0,1] для typical images
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        vec = np.clip(vec, 0.0, 1.0)
        return vec
