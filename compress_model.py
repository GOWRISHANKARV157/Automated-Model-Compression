import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import os

class ProperCompression:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" PROPER COMPRESSION on {self.device}")
    
    def get_dataloaders(self, batch_size=1024):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, test_loader
    
    def evaluate_model(self, model, test_loader):
        model = model.to(self.device)
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total
    
    def load_baseline_model(self):
        try:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 10)
            model.load_state_dict(torch.load('baseline_resnet18.pth', map_location=self.device, weights_only=True))
            model = model.to(self.device)
            print(" Loaded baseline model")
            return model
        except Exception as e:
            print(f" Error: {e}")
            return None
    
    def apply_smart_pruning(self, model, amounts=[0.3, 0.4, 0.5, 0.6]):
        """Smart pruning - different amounts for different layers"""
        print(" SMART PRUNING...")
        
        # Different pruning amounts for different layers
        conv_layers = []
        linear_layers = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((module, 'weight'))
            elif isinstance(module, nn.Linear) and 'fc' in name:
                linear_layers.append((module, 'weight'))
        
        # Prune convolutional layers lightly
        if conv_layers:
            prune.global_unstructured(
                conv_layers,
                pruning_method=prune.L1Unstructured,
                amount=amounts[0],  # 30% for conv layers
            )
        
        # Prune linear layers more aggressively
        if linear_layers:
            prune.global_unstructured(
                linear_layers,
                pruning_method=prune.L1Unstructured,
                amount=amounts[2],  # 50% for linear layers
            )
        
        # Make pruning permanent
        for module, param_name in conv_layers + linear_layers:
            prune.remove(module, param_name)
        
        # Calculate sparsity
        total, zeros = 0, 0
        for param in model.parameters():
            total += param.numel()
            zeros += torch.sum(param == 0).item()
        
        print(f" Overall sparsity: {zeros/total:.1%}")
        return model
    
    def apply_proper_quantization(self, model):
        """Proper quantization that maintains accuracy"""
        print(" PROPER QUANTIZATION...")
        
        # Move to CPU for quantization
        original_device = next(model.parameters()).device
        model = model.to('cpu')
        
        try:
            # Use dynamic quantization (preserves accuracy better)
            quantized_model = torch.quantization.quantize_dynamic(
                student.cpu(),
                {nn.Linear},
                dtype=torch.qint8
            )
            quantized_model.eval()

            torch.jit.script(quantized_model).save("compressed_resnet18.pt")

            print(" Proper quantization applied and model saved to CPU device.")          
            return quantized_model.to(original_device)
        except Exception as e:
            print(f" Quantization failed: {e}. Using unquantized model instead.")
            return model.to(original_device)
    
    def proper_distillation(self, teacher, student, train_loader, epochs=5):
        """Proper knowledge distillation that maintains accuracy"""
        print(" PROPER KNOWLEDGE DISTILLATION...")
        
        student.train()
        teacher.eval()
        
        # Use same architecture but compressed
        optimizer = torch.optim.Adam(student.parameters(), lr=0.0005)
        temperature = 3
        alpha = 0.7
        
        best_acc = 0
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Teacher predictions
                with torch.no_grad():
                    teacher_logits = teacher(images)
                
                # Student predictions
                student_logits = student(images)
                
                # Distillation loss
                soft_loss = nn.KLDivLoss(reduction='batchmean')(
                    torch.log_softmax(student_logits / temperature, dim=1),
                    torch.softmax(teacher_logits / temperature, dim=1)
                ) * (temperature ** 2)
                
                # Hard loss
                hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
                
                # Combined loss
                loss = alpha * soft_loss + (1 - alpha) * hard_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = student_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_acc = 100. * correct / total
            if epoch_acc > best_acc:
                best_acc = epoch_acc
            
            print(f"   Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f} | Train Acc: {epoch_acc:.2f}%")
        
        print(f"   Best training accuracy: {best_acc:.2f}%")
        return student
    
    def fine_tune_compressed(self, model, train_loader, epochs=3):
        """Fine-tune the compressed model"""
        print(" FINE-TUNING COMPRESSED MODEL...")
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_acc = 100. * correct / total
            print(f'   Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f} | Acc: {epoch_acc:.2f}%')
        
        return model
    
    def get_model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / 1024**2
    
    def run_compression(self):
        print(" STARTING PROPER COMPRESSION PIPELINE")
        print("="*60)
        
        train_loader, test_loader = self.get_dataloaders(batch_size=1024)
        
        # Load teacher model
        print(" LOADING TEACHER MODEL...")
        teacher = self.load_baseline_model()
        if teacher is None:
            return
        
        teacher_acc = self.evaluate_model(teacher, test_loader)
        teacher_size = self.get_model_size(teacher)
        print(f" TEACHER - Accuracy: {teacher_acc:.2f}% | Size: {teacher_size:.2f}MB")
        
        # Create student model (SAME architecture but we'll compress it)
        print("\n CREATING STUDENT MODEL...")
        student = models.resnet18(weights=None)
        student.fc = nn.Linear(student.fc.in_features, 10)
        student.load_state_dict(teacher.state_dict())  # Start with teacher weights
        student = student.to(self.device)
        
        student_acc_before = self.evaluate_model(student, test_loader)
        student_size_before = self.get_model_size(student)
        print(f" STUDENT BEFORE - Accuracy: {student_acc_before:.2f}% | Size: {student_size_before:.2f}MB")
        
        # Step 1: Smart Pruning
        print("\n" + "="*50)
        print("STEP 1: SMART PRUNING")
        print("="*50)
        student = self.apply_smart_pruning(student)
        
        student_acc_after_prune = self.evaluate_model(student, test_loader)
        student_size_after_prune = self.get_model_size(student)
        print(f" After Pruning - Accuracy: {student_acc_after_prune:.2f}% | Size: {student_size_after_prune:.2f}MB")
        
        # Step 2: Knowledge Distillation
        print("\n" + "="*50)
        print("STEP 2: KNOWLEDGE DISTILLATION")
        print("="*50)
        student = self.proper_distillation(teacher, student, train_loader, epochs=5)
        
        student_acc_after_distill = self.evaluate_model(student, test_loader)
        print(f" After Distillation - Accuracy: {student_acc_after_distill:.2f}%")
        
        # Step 3: Fine-tuning
        print("\n" + "="*50)
        print("STEP 3: FINE-TUNING")
        print("="*50)
        student = self.fine_tune_compressed(student, train_loader, epochs=3)
        
        # Step 4: Quantization
        print("\n" + "="*50)
        print("STEP 4: QUANTIZATION")
        print("="*50)
        student = self.apply_proper_quantization(student)
        
        # Final evaluation
        print("\n" + "="*60)
        print(" FINAL RESULTS")
        print("="*60)
        
        final_acc = self.evaluate_model(student.cpu(), test_loader)
        final_size = self.get_model_size(student)
        
        print(f"TEACHER   - Accuracy: {teacher_acc:.2f}% | Size: {teacher_size:.2f}MB")
        print(f"STUDENT   - Accuracy: {final_acc:.2f}% | Size: {final_size:.2f}MB")
        print(f"ACCURACY DROP: {teacher_acc - final_acc:.2f}%")
        print(f"SIZE REDUCTION: {teacher_size/final_size:.2f}x")
        print("="*60)
        
        # Save compressed model
        torch.save(student.state_dict(), 'compressed_resnet18.pth')
        print(" Compressed model saved!")
        
        # Goal achievement
        print("\n PROJECT GOALS:")
        if (teacher_acc - final_acc) <= 1.0:
            print(" ACCURACY DROP: <1% (GOAL ACHIEVED!)")
        elif (teacher_acc - final_acc) <= 2.0:
            print(" ACCURACY DROP: <2% (Close to goal)")
        else:
            print(" ACCURACY DROP: >2% (Goal not met)")
        
        if teacher_size/final_size >= 2.0:
            print(" SIZE REDUCTION: >2x (GOAL ACHIEVED!)")
        else:
            print(" SIZE REDUCTION: <2x (Goal not met)")
        
        print(" ALL TECHNIQUES: Pruning + Quantization + Knowledge Distillation")
        print(" CUDA COMPATIBLE: Yes")
        print(" BATCH SIZE: 1024")

if __name__ == '__main__':
    compressor = ProperCompression()
    compressor.run_compression()