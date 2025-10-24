import numpy as np
import cv2
from pathlib import Path
from scipy import ndimage
from skimage import exposure, filters, measure, feature
import warnings
warnings.filterwarnings('ignore')

class ImageQualityAnalyzer:
    """Image quality analyzer for RL preprocessing system using OpenCV"""
    
    def __init__(self):
        self.quality_thresholds = {
            'brightness_ideal_range': (0.4, 0.6),
            'contrast_min': 0.15,
            'edge_min': 0.03,
            'noise_max': 0.08,
            'blur_threshold': 100.0,
            'saturation_min': 0.1,
            'entropy_min': 4.0,
            'color_balance_max_bias': 0.15,
            'exposure_well_exposed_range': (0.1, 0.9)
        }
    
    def analyze_image(self, image_path):
        """Complete image analysis for RL agent using OpenCV"""
        try:
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                print(f"❌ Failed to load image: {image_path}")
                return None
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            basic_metrics = self._get_basic_metrics(img_rgb)
            quality_metrics = self._calculate_quality_metrics(img_rgb)
            problem_detection = self._detect_problems(quality_metrics)
            rl_features = self._prepare_rl_features(basic_metrics, quality_metrics, problem_detection)
            
            return {
                **basic_metrics,
                **quality_metrics,
                **problem_detection,
                **rl_features
            }
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return None
    
    def _get_basic_metrics(self, img_rgb):
        """Basic image metrics"""
        height, width = img_rgb.shape[:2]
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': round(width / height, 3),
            'total_pixels': width * height,
            'channels': img_rgb.shape[2] if len(img_rgb.shape) == 3 else 1
        }
    
    def _calculate_quality_metrics(self, img_rgb):
        """Calculate quality metrics using OpenCV"""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 255.0
        
        edge_score = self._calculate_robust_edge_score(gray)
        noise_score = self._estimate_robust_noise(gray)
        blur_score = self._detect_blur_opencv(gray)
        entropy_score = self._calculate_entropy_opencv(gray)
        
        color_metrics = self._calculate_color_metrics(img_rgb)
        
        # Exposure and dynamic range metrics
        exposure_metrics = self._calculate_exposure_metrics(gray)
        
        quality_score = self._calculate_overall_quality(
            brightness, contrast, edge_score, noise_score, blur_score, entropy_score,
            color_metrics, exposure_metrics
        )
        
        return {
            'quality_score': round(quality_score, 4),
            'brightness_score': round(brightness, 4),
            'contrast_score': round(contrast, 4),
            'edge_score': round(edge_score, 4),
            'noise_score': round(noise_score, 4),
            'blur_score': round(blur_score, 4),
            'entropy_score': round(entropy_score, 4),
            **color_metrics,
            **exposure_metrics
        }
    
    def _calculate_color_metrics(self, img_rgb):
        """Color metrics for RL"""
        try:
            # Convert to HSV for saturation and hue analysis
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            saturation = np.mean(img_hsv[:, :, 1]) / 255.0
            hue_variance = np.std(img_hsv[:, :, 0]) / 180.0
            
            # Color balance and dominant channels
            mean_b, mean_g, mean_r = np.mean(img_rgb, axis=(0, 1)) / 255.0
            color_balance_bias = np.std([mean_r, mean_g, mean_b])  # Color imbalance
            
            # Color problem detection
            is_color_cast = self._detect_color_cast(img_rgb)
            dominant_channel = np.argmax([mean_r, mean_g, mean_b])  # 0=R, 1=G, 2=B
            
            # Color contrast analysis (difference between colors)
            color_contrast = self._calculate_color_contrast(img_rgb)
            
            # Check for monochrome/sepia
            is_monochrome = self._check_monochrome(img_rgb, saturation)
            
            return {
                'saturation': round(saturation, 4),
                'hue_variance': round(hue_variance, 4),
                'color_balance_bias': round(color_balance_bias, 4),
                'color_contrast': round(color_contrast, 4),
                'is_color_cast': is_color_cast,
                'dominant_channel': dominant_channel,
                'is_monochrome': is_monochrome
            }
        except Exception as e:
            return {
                'saturation': 0.0, 'hue_variance': 0.0, 'color_balance_bias': 0.0,
                'color_contrast': 0.0, 'is_color_cast': 0, 'dominant_channel': -1,
                'is_monochrome': 0
            }
    
    def _calculate_exposure_metrics(self, gray):
        """Exposure and dynamic range metrics"""
        try:
            # Histogram for brightness distribution analysis
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Percentage of overexposed and underexposed pixels
            overexposed = np.sum(gray > 240) / gray.size
            underexposed = np.sum(gray < 15) / gray.size
            
            # Dynamic range (difference between 5% and 95% percentiles)
            p5, p95 = np.percentile(gray, [5, 95])
            dynamic_range = (p95 - p5) / 255.0
            
            # Contrast via histogram (entropy of brightness distribution)
            hist_entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            normalized_hist_entropy = hist_entropy / 6.0  # Normalization
            
            # HDR/high contrast detection
            is_high_contrast = 1 if dynamic_range > 0.8 else 0
            has_exposure_problems = 1 if (overexposed > 0.1 or underexposed > 0.1) else 0
            
            return {
                'overexposed_ratio': round(overexposed, 4),
                'underexposed_ratio': round(underexposed, 4),
                'dynamic_range': round(dynamic_range, 4),
                'histogram_entropy': round(normalized_hist_entropy, 4),
                'is_high_contrast': is_high_contrast,
                'has_exposure_problems': has_exposure_problems
            }
        except:
            return {
                'overexposed_ratio': 0.0, 'underexposed_ratio': 0.0, 'dynamic_range': 0.0,
                'histogram_entropy': 0.0, 'is_high_contrast': 0, 'has_exposure_problems': 0
            }
    
    def _detect_color_cast(self, img_rgb):
        """Color cast detection"""
        try:
            # Channel mean values
            mean_b, mean_g, mean_r = np.mean(img_rgb, axis=(0, 1))
            
            # If one channel significantly dominates - possible color cast
            max_mean = max(mean_r, mean_g, mean_b)
            min_mean = min(mean_r, mean_g, mean_b)
            
            # Strong imbalance = color cast
            return 1 if (max_mean - min_mean) > 50 else 0
        except:
            return 0
    
    def _calculate_color_contrast(self, img_rgb):
        """Calculate color contrast (color diversity)"""
        try:
            # Laplacian in color space for color boundary assessment
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            color_edges = cv2.Laplacian(l_channel, cv2.CV_64F).var() / 1000.0
            return min(color_edges, 1.0)
        except:
            return 0.0
    
    def _check_monochrome(self, img_rgb, saturation):
        """Check for monochrome image"""
        try:
            # If saturation is very low and colors are close to each other
            std_rgb = np.std(img_rgb, axis=(0, 1))
            avg_std = np.mean(std_rgb)
            return 1 if (saturation < 0.1 and avg_std < 20) else 0
        except:
            return 0
    
    def _calculate_robust_edge_score(self, gray):
        """Sharpness assessment via OpenCV"""
        try:
            # Method 1: Sobel filter
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
            edge1 = np.mean(sobel_magnitude) / 255.0
            
            # Method 2: Canny edge detector
            edges_canny = cv2.Canny(gray, 50, 150)
            edge2 = np.mean(edges_canny) / 255.0
            
            # Method 3: Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
            
            return (edge1 + edge2 + laplacian_var) / 3.0
        except:
            return 0.0
    
    def _estimate_robust_noise(self, gray):
        """Noise estimation via multiple OpenCV methods"""
        try:
            # Method 1: Median filter + difference
            median_filtered = cv2.medianBlur(gray, 3)
            noise_residual = cv2.absdiff(gray, median_filtered)
            noise1 = np.mean(noise_residual) / 255.0
            
            # Method 2: Gaussian blur + difference
            gaussian_filtered = cv2.GaussianBlur(gray, (5, 5), 0)
            high_freq = cv2.absdiff(gray, gaussian_filtered)
            noise2 = np.mean(high_freq) / 255.0
            
            # Method 3: Variation in small blocks
            h, w = gray.shape
            block_size = 8
            variances = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    variances.append(np.var(block))
            noise3 = np.mean(variances) / 255.0 if variances else 0.0
            
            return (noise1 + noise2 + noise3) / 3.0
        except:
            return 0.0
    
    def _detect_blur_opencv(self, gray):
        """Blur detection via OpenCV"""
        try:
            # Method 1: Variance of Laplacian
            fm_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Method 2: FFT-based blur detection
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)
            
            # FFT central region for high-frequency component assessment
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            fft_center = magnitude_spectrum[cy-10:cy+10, cx-10:cx+10]
            fft_mean = np.mean(fft_center)
            
            # Combined blur assessment
            blur_score_laplacian = min(fm_laplacian / 500.0, 1.0)
            blur_score_fft = min(fft_mean / 50.0, 1.0)
            
            return (blur_score_laplacian + blur_score_fft) / 2.0
        except:
            return 0.0
    
    def _calculate_entropy_opencv(self, gray):
        """Image entropy via OpenCV histogram"""
        try:
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist[hist > 0]
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log2(prob))
            return entropy / 8.0
        except:
            return 0.0
    
    def _calculate_overall_quality(self, brightness, contrast, edge, noise, blur, entropy, color_metrics, exposure_metrics):
        """Overall quality score considering all metrics"""
        brightness_score = 1 - abs(brightness - 0.5)
        contrast_score = min(contrast * 2.5, 1.0)
        edge_score = min(edge * 4, 1.0)
        noise_score = 1 - min(noise * 8, 1.0)
        blur_score = blur
        entropy_score = min(entropy / 6.0, 1.0)
        
        # Color metrics
        saturation_score = min(color_metrics.get('saturation', 0) * 2, 1.0)
        color_balance_score = 1 - min(color_metrics.get('color_balance_bias', 0) * 3, 1.0)
        
        # Exposure metrics
        exposure_score = 1 - min(exposure_metrics.get('overexposed_ratio', 0) * 5 + 
                                exposure_metrics.get('underexposed_ratio', 0) * 5, 1.0)
        dynamic_range_score = exposure_metrics.get('dynamic_range', 0)
        
        weights = [0.10, 0.15, 0.15, 0.10, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08]
        scores = [brightness_score, contrast_score, edge_score, noise_score, blur_score, 
                 entropy_score, saturation_score, color_balance_score, exposure_score, dynamic_range_score]
        
        return np.average(scores, weights=weights)
    
    def _detect_problems(self, metrics):
        """Detection of specific problems for RL agent"""
        return {
            'needs_contrast_boost': 1 if metrics['contrast_score'] < self.quality_thresholds['contrast_min'] else 0,
            'needs_brightness_fix': 1 if not (self.quality_thresholds['brightness_ideal_range'][0] <= metrics['brightness_score'] <= self.quality_thresholds['brightness_ideal_range'][1]) else 0,
            'needs_sharpening': 1 if metrics['edge_score'] < self.quality_thresholds['edge_min'] else 0,
            'needs_denoising': 1 if metrics['noise_score'] > self.quality_thresholds['noise_max'] else 0,
            'needs_deblurring': 1 if metrics['blur_score'] < 0.3 else 0,
            'needs_saturation_boost': 1 if metrics.get('saturation', 0) < self.quality_thresholds['saturation_min'] else 0,
            'needs_color_balance': 1 if metrics.get('color_balance_bias', 0) > self.quality_thresholds['color_balance_max_bias'] else 0,
            'needs_exposure_fix': 1 if metrics.get('has_exposure_problems', 0) else 0,
            'is_low_entropy': 1 if metrics['entropy_score'] < self.quality_thresholds['entropy_min'] else 0,
            'has_color_cast': 1 if metrics.get('is_color_cast', 0) else 0
        }
    
    def _prepare_rl_features(self, basic_metrics, quality_metrics, problem_detection):
        """Prepare features for RL agent"""
        # Quality categorization
        quality_level = 0  # low
        if quality_metrics['quality_score'] > 0.7:
            quality_level = 2  # high
        elif quality_metrics['quality_score'] > 0.4:
            quality_level = 1  # medium
        
        # Problem priority
        problem_priority = sum(problem_detection.values())
        
        # Image type and size
        image_type = 1 if basic_metrics['channels'] == 3 else 0
        image_size_category = 0  # small
        if basic_metrics['total_pixels'] > 1000000:
            image_size_category = 2  # large
        elif basic_metrics['total_pixels'] > 250000:
            image_size_category = 1  # medium
        
        # Main problems (for prioritization)
        critical_problems = (
            problem_detection['needs_exposure_fix'] +
            problem_detection['needs_deblurring'] +
            problem_detection['has_color_cast']
        )
        
        return {
            'quality_level': quality_level,
            'problem_priority': problem_priority,
            'critical_problems': critical_problems,
            'needs_any_processing': 1 if problem_priority > 0 else 0,
            'image_type': image_type,
            'image_size_category': image_size_category,
            'is_high_contrast': quality_metrics.get('is_high_contrast', 0),
            'is_monochrome': quality_metrics.get('is_monochrome', 0)
        }