import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime

from src.detection.detector import GroundingDINODetector
from src.models.baselines import MODEL_BUILDERS


class WasteDetectionClassificationPipeline:
    """
    Complete waste detection and classification pipeline
    Implements the desired workflow:
    Input Image → GroundingDINO Detector → Bounding Boxes + Crops → Classifier → Results
    """
    
    def __init__(self, detector: GroundingDINODetector, classifier_model, 
                 class_names: List[str], device: str):
        """
        Initialize pipeline
        
        Args:
            detector: CVDetector detector instance
            classifier_model: Trained waste classification model
            class_names: List of waste class names
            device: Device to run inference on
        """
        self.detector = detector
        self.classifier = classifier_model
        self.class_names = class_names
        self.device = device
        
        print(f"Pipeline initialized on device: {device}")
        print(f"Class names: {class_names}")
    
    def process_image(self, image_path: str,
                     classification_confidence: float = 0.5) -> Dict:
        """
        Process single image through complete pipeline
        
        Args:
            image_path: Path to input image
            classification_confidence: Minimum confidence for classification
            
        Returns:
            Dictionary containing complete results
        """
        print(f"🔄 Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Step 1: Grounding CV Detector
        print("Step 1: Detecting objects with CV Detector...")
        detections, annotated_image, labels = self.detector.detect(image)
        
        # Step 2: Extract object crops
        print("Step 2: Extracting object crops...")
        crops_with_bbox = self.detector.extract_object_crops(image, detections)
        
        print(f"Found {len(crops_with_bbox)} potential waste objects")
        
        # Step 3: Classify each crop
        print("Step 3: Classifying objects...")
        classification_results = []
        
        for i, (crop, bbox) in enumerate(crops_with_bbox):
            try:
                # Preprocess crop for classification
                input_tensor = self.detector.preprocess_crop(crop)
                
                # Classify
                with torch.no_grad():
                    outputs = self.classifier(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    confidence_value = confidence.item()
                    predicted_class = predicted.item()
                    class_name = self.class_names[predicted_class]
                    
                    if confidence_value >= classification_confidence:
                        result = {
                            'object_id': int(i + 1),
                            'bbox': bbox.tolist() if isinstance(bbox, np.ndarray) else list(map(int, bbox)),
                            'class': class_name,
                            'class_id': int(predicted_class),  
                            'confidence': float(confidence_value), 
                        }

                        classification_results.append(result)
                        
                        print(f"   Object {i+1}: {class_name} ({confidence_value:.3f})")
                    else:
                        print(f"   Object {i+1}: {class_name} ({confidence_value:.3f}) - LOW CONFIDENCE")
                        
            except Exception as e:
                print(f"❌ Error classifying object {i+1}: {e}")
                continue
        
        # Step 4: Prepare final results
        final_results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'detection_stats': {
                'total_detected': len(crops_with_bbox),
                'total_classified': len(classification_results),
                'classification_confidence_threshold': classification_confidence
            },
            'objects': classification_results,
            'class_distribution': self._calculate_class_distribution(classification_results)
        }
        
        print(f"Processing complete: {len(classification_results)} objects classified")
        return final_results
    
    def _calculate_class_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of classified objects"""
        distribution = {}
        for result in results:
            class_name = result['class']
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def visualize_results(self, image_path: str, results: Dict, 
                         save_path: Optional[str] = None, 
                         show: bool = True) -> np.ndarray:
        """
        Visualize detection and classification results
        
        Args:
            image_path: Path to original image
            results: Results dictionary from process_image
            save_path: Path to save visualization
            show: Whether to display the image
            
        Returns:
            Annotated image
        """
        # Load image
        image_path = os.path.normpath(image_path)  # нормализация пути
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}. "
                             f"Check if file exists and path is correct.")

        # Create copy for annotation
        annotated_image = image.copy()
        
        # Draw bounding boxes and labels
        for obj in results['objects']:
            bbox = obj['bbox']
            class_name = obj['class']
            confidence = obj['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = self._get_color_for_class(class_name)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(annotated_image, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add statistics text
        stats_text = f"Objects: {results['detection_stats']['total_classified']}/{results['detection_stats']['total_detected']}"
        cv2.putText(annotated_image, stats_text,
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Save or display
        # if save_path:
        #     cv2.imwrite(save_path, annotated_image)
        #     print(f"Visualization saved to: {save_path}")
        
        if show:
            cv2.imshow("Waste Detection & Classification Results", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated_image
    
    def _get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """Get consistent color for each waste class"""
        color_map = {
            'plastic': (255, 0, 0),      # Red
            'paper': (0, 255, 0),        # Green  
            'cardboard': (0, 0, 255),    # Blue
            'metal': (255, 255, 0),      # Cyan
            'glass': (255, 0, 255),      # Magenta
            'trash': (0, 255, 255),      # Yellow
            'battery': (255, 165, 0),    # Orange
            'clothes': (128, 0, 128),    # Purple
            'biological': (165, 42, 42)  # Brown
        }
        return color_map.get(class_name, (128, 128, 128))  # Gray for unknown
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save results to JSON file
        
        Args:
            results: Results dictionary
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")


def create_complete_pipeline(device: str = None) -> WasteDetectionClassificationPipeline:
    """
    Factory function to create complete waste detection and classification pipeline
    
    Args:
        device: Device to run inference on
        
    Returns:
        Initialized pipeline instance
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing pipeline on device: {device}")
    
    # Define class names (should match your training)
    class_names = ['cardboard', 'paper', 'plastic', 'metal', 'glass', 
                   'trash', 'battery', 'clothes', 'biological']
    
    # Step 1: Load classifier model
    print("Step 1: Loading classifier model...")

    model_path = Path(__file__).resolve().parent.parent / "models" / "baseline.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Укажи свою архитектуру!
    model_name = "resnet18"

    # Создаём пустую модель
    classifier = MODEL_BUILDERS[model_name](
        num_classes=len(class_names),
        pretrained=False
    )

    # Загружаем веса
    state_dict = torch.load(model_path, map_location=device)
    classifier.load_state_dict(state_dict)

    classifier.to(device)
    classifier.eval()

    print(f"Classifier loaded: {model_name}")

    
    # Step 2: Initialize detector
    print("Step 2: Initializing detector...")
    detector = GroundingDINODetector(device=device)
    
    # Step 3: Create pipeline
    print("Step 3: Creating pipeline...")
    pipeline = WasteDetectionClassificationPipeline(
        detector=detector,
        classifier_model=classifier,
        class_names=class_names,
        device=device
    )
    
    print("Complete waste detection pipeline initialized successfully!")
    return pipeline