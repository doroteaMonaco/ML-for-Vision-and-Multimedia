# Machine Learning for Visual and Multimedia - Labs

This repository contains the laboratory materials for the **Machine Learning for Visual and Multimedia** course at Politecnico di Torino (Master's Degree, Second Year).

## Contents

### 📚 LAB1: MNIST Classification with PyTorch
- **Main File**: `mnist_classification_pytorch.ipynb`
- **Description**: Classification of handwritten digits (MNIST) using Convolutional Neural Networks (CNN) in PyTorch
- **Topics**: Dataset loading, preprocessing, CNN training, model evaluation

### 🧮 LAB1: PyTorch Gradient Computation
- **Main File**: `pytorch_gradient_computation.ipynb`
- **Description**: Gradient computation and backpropagation in PyTorch
- **Topics**: Autograd, derivative calculation, optimization

### ⚙️ LAB2: Hyperparameter Optimization
- **Main File**: `Hyperparameter_Optimization.ipynb`
- **Description**: Hyperparameter optimization techniques for neural networks
- **Topics**: Grid search, random search, cross-validation, parameter tuning

### 🔄 LAB3: Transfer Learning
- **Main File**: `Transfer Learning.ipynb` / `Transfer Learning-Dorotea.ipynb`
- **Description**: Transfer learning application for image classification
- **Topics**: Pre-trained models, fine-tuning, feature extraction

### 🎯 LAB4: GradCAM and Visualization
- **GradCAM**: `GradCAM_Pytorch_CatsvsDogs.ipynb`
  - Gradient-weighted Class Activation Mapping for neural network interpretation
  - Visualization of important regions in predictions
  - Cats vs Dogs classification
  
- **TensorBoard and t-SNE**: `TensorBoardProjector_tSNE_embedding_Pytorch_CatsvsDogs (1).ipynb`
  - Embedding visualization with t-SNE
  - Training monitoring with TensorBoard

### 🔬 LAB5: Advanced Models
- **Main Files**: `LAB5.ipynb` / `Lab5_AdvancedModels_slideExamples.ipynb`
- **Description**: Advanced models and deep learning techniques
- **Topics**: Modern architectures, data augmentation, regularization techniques

### 🎵 LAB6: Audio Keyword Spotting
- **Main File**: `ml4vmm26_lab06_audio_keyword_spotting_assignment.ipynb`
- **Description**: Keyword spotting in audio signals
- **Topics**: Audio processing, spectrograms, CNN for audio classification

### 🎄 LAB7: Christmas GAN
- **Main File**: `Lab7_Christmas_GAN_Pytorch_traccia.ipynb`
- **Description**: Generative Adversarial Network (GAN) for generating Christmas tree images
- **Topics**: GAN architecture, generator/discriminator training, image generation, adversarial learning

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- TensorBoard
- librosa (for audio processing)

Install dependencies with:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn tensorboard librosa
```

## How to Use

1. Clone the repository
2. Install the dependencies
3. Open the Jupyter notebooks in order (LAB1 → LAB7)
4. Follow the instructions and exercises within each notebook

## Directory Structure

```
Labs/
├── LAB1/
│   ├── mnist_classification_pytorch.ipynb
│   └── pytorch_gradient_computation.ipynb
├── LAB2/
│   ├── Hyperparameter_Optimization.ipynb
│   └── README.md
├── LAB3/
│   ├── Transfer Learning.ipynb
│   └── Transfer Learning-Dorotea.ipynb
├── LAB4/
│   ├── GradCAM_Pytorch_CatsvsDogs.ipynb
│   └── TensorBoardProjector_tSNE_embedding_Pytorch_CatsvsDogs (1).ipynb
├── LAB5/
│   ├── Lab5_AdvancedModels_slideExamples.ipynb
│   └── LAB5.ipynb
├── LAB6/
│   └── ml4vmm26_lab06_audio_keyword_spotting_assignment.ipynb
├── LAB7/
│   └── Lab7_Christmas_GAN_Pytorch_traccia.ipynb
└── README.md
```

## Important Notes

- Some notebooks require large datasets that may need to be downloaded
- It is recommended to use a GPU to accelerate model training
- Each notebook contains detailed comments and explanations

## Author

Dorotea Monaco - Politecnico di Torino

## License

This material is provided for educational purposes.
