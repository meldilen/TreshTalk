# rl/env.py
from typing import Tuple, Dict, Any, Optional, List
import gym
from gym import spaces
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat
import random
import os
import tempfile
import math
import shutil

# Небольшая утилита для вычисления простых признаков изображения.
def compute_image_features(image_path: str) -> Dict[str, float]:
    """
    Возвращает словарь признаков, соответствующих state_features.
    Эти признаки — простые, быстрые к расчёту: яркость, контраст (approx), энтропия,
    доля пересветов/недосветов, количество краёв (через градиент).
    Для лучшей работы можно заменить на CNN-эмбеддинги или histogram-of-gradients.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0  # H x W x C, 0..1

    # brightness: среднее по яркости (luma)
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    brightness_score = float(np.clip(lum.mean(), 0.0, 1.0))

    # contrast: std of luma
    contrast_score = float(np.clip(lum.std(), 0.0, 1.0))

    # saturation (approx): std across channels normalized
    saturation = float(np.clip(arr.std(axis=(0, 1)).mean(), 0.0, 1.0))

    # entropy (per-channel mean entropy)
    def channel_entropy(channel):
        # approximate using histogram
        hist, _ = np.histogram(channel, bins=256, range=(0, 1))
        p = hist / (hist.sum() + 1e-8)
        p = p[p > 0]
        return -float(np.sum(p * np.log2(p)))
    entropy = (channel_entropy(arr[..., 0]) + channel_entropy(arr[..., 1]) + channel_entropy(arr[..., 2])) / 3.0
    # normalize entropy to 0..1 (max entropy ~8 for 256 bins)
    entropy_norm = float(np.clip(entropy / 8.0, 0.0, 1.0))

    # edges: simple gradient magnitude mean (Sobel-free simple filter)
    gx = np.abs(np.diff(lum, axis=1)).mean()
    gy = np.abs(np.diff(lum, axis=0)).mean()
    edge_score = float(np.clip((gx + gy) / 2.0 * 10.0, 0.0, 1.0))  # scaled approx

    # over/underexposed ratio
    over = float(np.mean(lum > 0.95))
    under = float(np.mean(lum < 0.05))

    # noise/blur proxies: laplacian variance ~ focus measure
    try:
        lap = np.var(lum - cv2.GaussianBlur(lum, (3, 3), 0))  # requires opencv if available
        blur_score = float(np.clip(1.0 / (1.0 + lap), 0.0, 1.0))
    except Exception:
        # если нет cv2, используем simple proxy: inverse of edge_score
        blur_score = float(np.clip(1.0 - edge_score, 0.0, 1.0))

    # assemble features mapping to state_features in env
    features = {
        "quality_score": float((entropy_norm + contrast_score + edge_score) / 3.0),
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "edge_score": edge_score,
        "noise_score": 0.0,  # placeholder (could compute with wavelets)
        "blur_score": blur_score,
        "saturation": saturation,
        "color_balance_bias": 0.0,  # placeholder: could use mean channel differences
        "overexposed_ratio": over,
        "underexposed_ratio": under,
        "dynamic_range": float(np.clip(lum.max() - lum.min(), 0.0, 1.0)),
        "is_low_entropy": float(entropy_norm < 0.2),
        "has_color_cast": 0.0
    }
    return features


class WasteRLMultiStepEnv(gym.Env):
    """
    Gym-совместимая среда для RL, которая по шагам применяет предобработку к изображению,
    пересчитывает признаки и возвращает обновлённые наблюдения.

    Observation: numpy array shape (len(state_features),) с нормированными признаками 0..1
    Action: discrete выбирает одну из предопределённых трансформаций
    Episode: несколько действий (до max_steps) — затем терминальный сигнал (terminated=True)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 manifest_df,
                 model,
                 device,
                 class2idx: dict,
                 data_root: str,
                 max_steps: int = 5):
        super().__init__()
        self.df = manifest_df.reset_index(drop=True)  # pandas DataFrame
        self.model = model
        self.device = device
        self.class2idx = class2idx
        self.data_root = data_root
        self.max_steps = int(max_steps)

        # features we compute & return as observation
        self.state_features = [
            "quality_score", "brightness_score", "contrast_score", "edge_score",
            "noise_score", "blur_score", "saturation", "color_balance_bias",
            "overexposed_ratio", "underexposed_ratio", "dynamic_range",
            "is_low_entropy", "has_color_cast"
        ]

        self.actions = [
            "none", "brighten", "darken", "contrast_boost", "sharpen", "deblur",
            "denoise", "saturation_boost", "color_balance", "exposure_fix", "crop_center"
        ]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(self.state_features),), dtype=np.float32)

        # episode state
        self.current_row = None              # pandas Series for the example
        self.original_image_path: Optional[str] = None
        self.current_image_path: Optional[str] = None  # path to the current (possibly temp) image
        self.true_label = None
        self.current_step = 0
        self._temp_files: List[str] = []    # чтобы чистить временные файлы

    # Gym-compatible reset signature (совместимо с Gym >=0.21)
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.current_step = 0
        row = self.df.sample(1).iloc[0]
        self.current_row = row.copy()
        self.true_label = row["unified_class"]
        self.original_image_path = os.path.join(self.data_root, row["file_path"])
        self.current_image_path = self.original_image_path

        # compute features from the actual image (not from manifest fields)
        feat = compute_image_features(self.current_image_path)
        obs = np.array([feat[f] for f in self.state_features], dtype=np.float32)

        info = {"step": self.current_step, "file_path": row["file_path"], "true_label": self.true_label}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Выполняет действие, применяет трансформацию к image, пересчитывает признаки.
        Возвращает: observation, reward, terminated, truncated, info
        """
        assert self.current_image_path is not None, "call reset() before step()"

        action_idx = int(action)
        action_name = self.actions[action_idx]
        self.current_step += 1

        # Оценка до действия
        old_correct, old_conf, _ = evaluate_single_image(self.model, self.device, self.current_image_path,
                                                         self.true_label, self.class2idx)

        # Применяем действие — получаем новый временный файл
        new_img_path = self._apply_action_to_image(self.current_image_path, action_name)

        # удаляем предыдущий временный файл (чтобы не накапливать), но не удаляем исходный файл
        if self.current_image_path and self.current_image_path != self.original_image_path:
            try:
                if os.path.exists(self.current_image_path):
                    os.remove(self.current_image_path)
            except Exception:
                pass

        self.current_image_path = new_img_path

        # Оценка после действия
        new_correct, new_conf, _ = evaluate_single_image(self.model, self.device, self.current_image_path,
                                                         self.true_label, self.class2idx)

        # reward: сочетание улучшения correctness и подъёма confidence, минус штраф за шаг
        reward = float((new_correct - old_correct) + (new_conf - old_conf))
        step_penalty = -0.01
        reward += step_penalty

        # terminated = мы достигли максимума шагов
        terminated = self.current_step >= self.max_steps
        truncated = False

        # recompute features from the new image to form next_state
        feat = compute_image_features(self.current_image_path)
        next_state = np.array([feat[f] for f in self.state_features], dtype=np.float32)

        info = {
            "step": self.current_step,
            "action": action_name,
            "reward": reward,
            "old_acc": old_correct,
            "new_acc": new_correct,
            "old_conf": float(old_conf),
            "new_conf": float(new_conf)
        }
        return next_state, reward, terminated, truncated, info

    def _apply_action_to_image(self, image_path: str, action: str) -> str:
        """
        Применяем действие к изображению и сохраняем во временный файл.
        Возвращаем путь к новому временному файлу.
        """
        img = Image.open(image_path).convert("RGB")

        if action == "none":
            out = img
        elif action == "brighten":
            out = ImageEnhance.Brightness(img).enhance(1.25)
        elif action == "darken":
            out = ImageEnhance.Brightness(img).enhance(0.8)
        elif action == "contrast_boost":
            out = ImageEnhance.Contrast(img).enhance(1.3)
        elif action == "sharpen":
            out = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        elif action == "deblur":
            out = img.filter(ImageFilter.UnsharpMask(radius=1, percent=100))
        elif action == "denoise":
            out = img.filter(ImageFilter.MedianFilter(size=3))
        elif action == "saturation_boost":
            out = ImageEnhance.Color(img).enhance(1.2)
        elif action == "color_balance":
            out = ImageEnhance.Color(img).enhance(1.05)
        elif action == "exposure_fix":
            out = ImageEnhance.Brightness(img).enhance(1.1)
        elif action == "crop_center":
            w, h = img.size
            cx, cy = w // 2, h // 2
            nw, nh = int(w * 0.9), int(h * 0.9)
            box = (cx - nw // 2, cy - nh // 2, cx + nw // 2, cy + nh // 2)
            out = img.crop(box).resize((w, h))
        else:
            out = img

        # Save to a temporary file
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp.name
        tmp.close()
        out.save(tmp_path, quality=95)
        self._temp_files.append(tmp_path)
        return tmp_path

    def close(self):
        # cleanup temp files
        for p in list(self._temp_files):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        self._temp_files = []
