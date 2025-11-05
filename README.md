# Automated Model Compression for Deep Neural Networks

A comprehensive implementation of automated model compression techniques including **Pruning, Quantization, and Knowledge Distillation** to reduce deep neural network size while maintaining accuracy.

## 🎯 Project Overview

This project demonstrates automated compression of ResNet-18 on CIFAR-10, achieving:
- **<1% accuracy drop** (from 82% to 81.5%)
- **>2x model size reduction** (from 42.7MB to 18.4MB)
- **CUDA-compatible** implementation
- **All three compression techniques** applied

## 📊 Results

| Metric | Original Model | Compressed Model | Improvement |
|--------|----------------|------------------|-------------|
| **Accuracy** | 82.0% | 81.5% | <1% drop  |
| **Model Size** | 42.7 MB | 18.4 MB | 2.32x reduction  |
| **Inference Time** | 3.45 ms | 3.12 ms | 1.11x faster ✅|
| **Techniques** | - | All Three Applied |  |

## 🛠️ Compression Techniques

### 1. Knowledge Distillation
- **Teacher Model**: Original ResNet-18
- **Student Model**: Compact architecture
- **Temperature**: 4.0
- **Loss**: Combined KL-Divergence + Cross-Entropy

### 2. Structured Pruning
- **Method**: Global L1 Unstructured Pruning
- **Sparsity**: 40% weights removed
- **Approach**: Fine-tuned after pruning

### 3. Quantization
- **Type**: Dynamic Quantization
- **Precision**: FP32 → INT8
- **Layers**: Linear and Convolutional

## 📁 Project Structure
MyModelCompressionProject/
├── train_baseline.py # Train original ResNet-18
├── compress_model.py # Main compression pipeline
├── evaluate_results.py # Compare models
├── utils.py # Utility functions
├── baseline_resnet18.pth # Original model weights
├── compressed_resnet18.pth # Compressed model weights
└── requirements.txt # Dependencies

## Run complete pipeline(Individual Execution)

# Step 1: Train baseline model
python train_baseline.py

# Step 2: Compress the model
python compress_model.py

# Step 3: Evaluate results
python evaluate_results.py

## 🔧 Configuration
Model Specifications:
Base Architecture: ResNet-18
Dataset: CIFAR-10 (10 classes)
Input Size: 32×32×3
Batch Size: 1024
Device: Automatic CUDA/CPU detection

Training Parameters:
Epochs: 8 (Baseline) + 10 (Distillation) + 5 (Fine-tuning)
Learning Rate: 0.001 (Baseline), 0.0001 (Fine-tuning)
Optimizer: Adam
Loss Function: Cross-Entropy + KL-Divergence

## 📈 Performance Metrics

Accuracy Preservation--
Original Model:  82.0% accuracy
Compressed Model: 81.5% accuracy  
Accuracy Drop: 0.5% (<1% target achieved)

Size Reduction--
Original Size:  42.7 MB
Compressed Size: 18.4 MB
Reduction Ratio: 2.32x (>2x target achieved)

## Hardware Compatibility
✅ NVIDIA CUDA Support

✅ CPU Fallback

✅ Batch Size Optimization

## 🎓 Technical Details

--Knowledge Distillation
# Temperature scaling
teacher_probs = softmax(teacher_logits / temperature)
student_probs = log_softmax(student_logits / temperature)
loss = α * KL_loss + (1-α) * CE_loss

--Pruning Strategy
# Global pruning
prune.global_unstructured(
    parameters,
    pruning_method=prune.L1Unstructured,
    amount=0.4
)

--Quantization Approach
# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)

## 🙏 Acknowledgments
PyTorch team for excellent deep learning framework

CIFAR-10 dataset providers

Research community for compression techniques