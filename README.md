# TreshTalk

An RL-powered system that selects optimal preprocessing operations to improve waste image classification accuracy.

---

### Overview

The goal of this project is to build an intelligent system that classifies waste items (plastic, glass, paper, metal, cardboard, trash) from images, while dynamically adapting the image preprocessing pipeline using reinforcement learning (RL). 

The assistant classifies waste items into appropriate categories and guides the user interactively via Telegram bot.
Instead of using a fixed preprocessing strategy (resize, normalize, etc.), we propose an RL agent that chooses a sequence of preprocessing operations (e.g., rotation, brightness adjustment, noise removal, cropping) to maximize the classification accuracy of a CNN model.

---

## State of the Art

**Key References:**
- **TrashNet Dataset**: Standard benchmark with 6 waste categories.
- **EfficientNet/ResNet**: SOTA for waste classification.
- **Interactive Assistants:** Limited prototypes exist, but none combine RL-based preprocessing with dialogue.

**Our Innovation:** RL-driven adaptive preprocessing + interactive chatbot guidance.

---

## Datasets

**Core Dataset:**  
TrashNet (2,500 images, 6 classes: glass, paper, cardboard, plastic, metal, trash)

**Extended Classes & Sources:**  
- **Batteries:** RecyBat24 (lithium-ion battery types)  
- **Lamps:** Adapted CIFAR-100 (lamp class → detection annotations)  
- **Industrial Objects:** WaRP (28 sub-categories for robustness testing)  
- **Generalization:** Garbage Classification (12 classes, 15K images) for broader coverage  

**Annotation Enhancements:**  
- Material properties (flexibility, transparency)  
- Surface conditions (food residues, damage)  
- Object size and geometric attributes  
- Synthetic data generation for rare classes  

---

## Success Metrics

| Metric | Target | Evaluation Method |
|--------|--------|-------------------|
| **Classification Accuracy** | ≥90% (general)<br>≥85% (hazardous) | Test set performance |
| **Preprocessing Improvement** | +3% over baseline | RL vs static pipeline A/B test |
| **Inference Speed** | <500 ms/image | GPU latency measurement |
| **Detection Performance** | mAP ≥0.85 | COCO evaluation metrics |
| **User Satisfaction** | ≥70% positive feedback | Telegram bot feedback system |
| **Deployment Ready** | Functional Telegram bot | End-to-end testing |

**Technical Benchmarks:**  
- Real-time capability: ≥10 FPS on edge devices  
- Robustness: Maintain performance under lighting/occlusion variations  
- Model Size: <500MB for deployment feasibility  

---

## Collaborators

- Dzhamilia Fatkullina (`d.fatkullina@innopolis.university`)
- Dziyana Melnikava (`dz.melnikava@innopolis.university`)