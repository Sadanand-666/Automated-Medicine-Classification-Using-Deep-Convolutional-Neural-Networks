# 🧠 High-Accuracy Medicine Image Classification Using VGG-19 and Advanced Data Augmentation
This project presents an **end-to-end deep learning pipeline** for **medicine image classification** using **Transfer Learning with VGG-19** in **PyTorch**.  
The system leverages **advanced Albumentations-based image augmentation**, a **custom dataset wrapper**, and a **fine-tuned classifier head** to achieve **high accuracy and strong generalization**.

## 🚀 Key Highlights

- Pretrained **VGG-19 (ImageNet weights)**
- Transfer Learning with frozen convolutional layers
- Advanced **Albumentations** data augmentation
- Custom PyTorch Dataset integration
- AdamW optimizer with learning-rate scheduling
- Extensive evaluation metrics & visualizations
- Final **Test Accuracy: 95.01%**

## 📂 Dataset Structure

medicine_data/
│
├── Train/

│ ├── Class_1/

│ ├── Class_2/

│ └── Class_N/

│

│ └── Test/

│ ├── Class_1/

│ ├── Class_2/

│ └── Class_N/

Each class directory contains labeled medicine images.

## ⚙️ Tech Stack

- **Language:** Python  
- **Deep Learning:** PyTorch, Torchvision  
- **Augmentation:** Albumentations, OpenCV  
- **Visualization:** Matplotlib, Seaborn  
- **Metrics:** Scikit-learn  
- **Hardware:** GPU (CUDA supported)  

## 🔄 Data Augmentation Pipeline

Albumentations was used to improve robustness and reduce overfitting.

**Augmentations Applied:**
- Resize & Random Crop  
- Horizontal Flip  
- Affine Transformations (scale, rotate, translate)  
- ImageNet Normalization  

## 🏗️ Model Architecture

**Backbone:** VGG-19 (Pretrained on ImageNet)

**Classifier Head:**
Linear (25088 → 512)
ReLU
Dropout (0.4)
Linear (512 → Number of Classes)


**Transfer Learning Strategy:**
- Convolutional layers frozen
- Only classifier layers trained


## ⚡ Training Configuration

| Parameter        | Value |
|------------------|-------|
| Optimizer        | AdamW |
| Learning Rate    | 0.0001 |
| Weight Decay     | 1e-4 |
| Loss Function    | CrossEntropyLoss |
| Scheduler        | StepLR (step=5, gamma=0.5) |
| Batch Size       | 32 |
| Epochs           | 25 |
| Device           | GPU / CPU |


## 📊 Training Results

- **Training Accuracy:** ~94%  
- **Validation Accuracy:** **95.01%**  
- Smooth convergence  
- Minimal overfitting  
- Stable loss reduction  


## 📈 Evaluation Metrics

The model was evaluated using:

- Accuracy & Loss curves  
- Smoothed learning curves  
- Confusion Matrix  
- Multi-class ROC-AUC curves (One-vs-Rest)  

These metrics provide both **global** and **class-wise** performance insights.

## 🔮 Future Improvements

Fine-tuning deeper VGG layers

Grad-CAM visualization for interpretability

EfficientNet / ResNet comparison

Cross-validation

Deployment via FastAPI or TorchServe

## 📜 Ethical & Usage Disclaimer

This project was developed with conceptual guidance and validation support from ChatGPT, while maintaining a clear understanding of:

CNN architectures

Transfer Learning principles

Data augmentation strategies

Model evaluation techniques

All experimentation, implementation, and analysis were conducted by the author.

# 👤 Author

Sadanand Bhandari

AI & Data Science Practitioner
