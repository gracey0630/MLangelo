import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
from huggingface_hub import hf_hub_download
import numpy as np


checkpoint_path = '/Users/gahyunyoon/Desktop/classes/eecs6893-final-project/best_model_stage2.pth'
data_dir = '/Users/gahyunyoon/Desktop/classes/eecs6893-final-project/data/reddit_data'
batch_size = 32
num_workers = 2

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class SuSyTransferLearning(nn.Module):
    def __init__(self, susy_model_path, freeze_backbone=True):
        super().__init__()
        
        self.susy_model = torch.jit.load(susy_model_path)
        
        if freeze_backbone:
            for param in self.susy_model.parameters():
                param.requires_grad = False
        
        self.projection = nn.Linear(6, 3)
    
    def forward(self, x):
        with torch.set_grad_enabled(self.training):
            susy_logits = self.susy_model(x)
        output = self.projection(susy_logits)
        return output


class SuSyOriginal(nn.Module):
    def __init__(self, susy_model_path):
        super().__init__()
        self.susy_model = torch.jit.load(susy_model_path)
    
    def forward(self, x):
        return self.susy_model(x)


class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        self.class_to_idx = {
            'Art': 0,
            'dalle2': 1,
            'midjourney': 2
        }
        
        self.samples = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist!")
                continue
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    self.samples.append((str(img_path), class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        print(f"Loaded {len(self.samples)} test images")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        class_counts = {name: 0 for name in self.class_to_idx.keys()}
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        for _, label in self.samples:
            class_counts[idx_to_class[label]] += 1
        
        print(f"  Class distribution: {class_counts}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            random_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(random_idx)


def evaluate_transfer_model(model, dataloader, device, class_names):
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_paths = []
    
    correct = 0
    total = 0
    
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    
    confusion_matrix = np.zeros((3, 3), dtype=int)
    
    print("\nEvaluating transfer learning model...")
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc='Testing Transfer'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_paths.extend(paths)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
                
                confusion_matrix[label][pred] += 1
    
    overall_accuracy = 100 * correct / total
    
    class_accuracies = {}
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            class_accuracies[name] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracies[name] = 0.0
    
    class_metrics = {}
    for i, name in enumerate(class_names):
        tp = confusion_matrix[i][i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        tn = total - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[name] = {
            'accuracy': class_accuracies[name],
            'precision': 100 * precision,
            'recall': 100 * recall,
            'f1': 100 * f1,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'count': class_total[i]
        }
    
    return {
        'overall_accuracy': overall_accuracy,
        'class_metrics': class_metrics,
        'confusion_matrix': confusion_matrix.tolist(),
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'paths': all_paths,
    }


def evaluate_original_model(model, dataloader, device):
    model.eval()
    
    original_class_names = ['authentic', 'dalle-3-images', 'diffusiondb', 
                           'midjourney-images', 'midjourney_tti', 'realisticSDXL']
    
    test_to_susy_mapping = {
        0: 0,
        1: 1,
        2: [3, 4]
    }
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    class_correct = {'Art': 0, 'dalle2': 0, 'midjourney': 0}
    class_total = {'Art': 0, 'dalle2': 0, 'midjourney': 0}
    
    susy_class_correct = {
        'authentic': 0,
        'dalle-3-images': 0,
        'midjourney-images': 0,
        'midjourney_tti': 0
    }
    susy_class_total = {
        'authentic': 0,
        'dalle-3-images': 0,
        'midjourney-images': 0,
        'midjourney_tti': 0
    }
    
    print("\nEvaluating original SuSy model (6 classes)...")
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc='Testing Original'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred = predicted[i].item()
                
                if true_label == 0:
                    test_class = 'Art'
                    class_total[test_class] += 1
                    susy_class_total['authentic'] += 1
                    
                    if pred == 0:
                        class_correct[test_class] += 1
                        susy_class_correct['authentic'] += 1
                
                elif true_label == 1:
                    test_class = 'dalle2'
                    class_total[test_class] += 1
                    susy_class_total['dalle-3-images'] += 1
                    
                    if pred == 1:
                        class_correct[test_class] += 1
                        susy_class_correct['dalle-3-images'] += 1
                
                elif true_label == 2:
                    test_class = 'midjourney'
                    class_total[test_class] += 1
                    
                    if pred == 3:
                        susy_class_total['midjourney-images'] += 1
                        if pred in [3, 4]:
                            class_correct[test_class] += 1
                        if pred == 3:
                            susy_class_correct['midjourney-images'] += 1
                    elif pred == 4:
                        susy_class_total['midjourney_tti'] += 1
                        if pred in [3, 4]:
                            class_correct[test_class] += 1
                        if pred == 4:
                            susy_class_correct['midjourney_tti'] += 1
                    else:
                        susy_class_total['midjourney-images'] += 1
    
    class_metrics = {}
    for class_name in ['Art', 'dalle2', 'midjourney']:
        total = class_total[class_name]
        correct = class_correct[class_name]
        
        recall = 100 * correct / total if total > 0 else 0
        accuracy = 100 * correct / total if total > 0 else 0
        misclassified = total - correct
        misclassified_pct = 100 * misclassified / total if total > 0 else 0
        
        class_metrics[class_name] = {
            'total': total,
            'correct': correct,
            'misclassified': misclassified,
            'accuracy': accuracy,
            'misclassified_percentage': misclassified_pct,
            'recall': recall
        }
    
    susy_class_metrics = {}
    for class_name in ['authentic', 'dalle-3-images', 'midjourney-images', 'midjourney_tti']:
        total = susy_class_total[class_name]
        correct = susy_class_correct[class_name]
        
        recall = 100 * correct / total if total > 0 else 0
        accuracy = 100 * correct / total if total > 0 else 0
        misclassified = total - correct
        misclassified_pct = 100 * misclassified / total if total > 0 else 0
        
        susy_class_metrics[class_name] = {
            'total': total,
            'correct': correct,
            'misclassified': misclassified,
            'accuracy': accuracy,
            'misclassified_percentage': misclassified_pct,
            'recall': recall
        }
    
    return {
        'class_metrics': class_metrics,
        'susy_class_metrics': susy_class_metrics,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'class_names': original_class_names
    }


def print_transfer_results(results, class_names):
    print("\nTEST RESULTS - TRANSFER LEARNING MODEL (3 Classes)")
    
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2f}%")
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Count':<8}")
    
    for name in class_names:
        metrics = results['class_metrics'][name]
        print(f"{name:<15} {metrics['accuracy']:>10.2f}% {metrics['precision']:>10.2f}% "
              f"{metrics['recall']:>10.2f}% {metrics['f1']:>10.2f}% {metrics['count']:>8}")
    
    print("\nConfusion Matrix (TP, FP, TN, FN per class):")
    for name in class_names:
        metrics = results['class_metrics'][name]
        print(f"\n{name}:")
        print(f"  True Positives (TP):  {metrics['tp']}")
        print(f"  False Positives (FP): {metrics['fp']}")
        print(f"  True Negatives (TN):  {metrics['tn']}")
        print(f"  False Negatives (FN): {metrics['fn']}")
    
    print("\nConfusion Matrix:")
    print(f"{'':>15} ", end="")
    for name in class_names:
        print(f"{name[:10]:>12}", end="")
    print()
    
    cm = results['confusion_matrix']
    for i, true_class in enumerate(class_names):
        print(f"{true_class:<15} ", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>12}", end="")
        print()


def print_original_results(results):
    print("\nTEST RESULTS - ORIGINAL SUSY MODEL (6 Classes)")
    
    print("\nPer-Class Performance (Test Dataset Classes):")
    print(f"{'Class':<15} {'Total':<10} {'Correct':<10} {'Misclassified':<15} {'Accuracy':<12} {'Recall':<12}")
    
    for class_name in ['Art', 'dalle2', 'midjourney']:
        metrics = results['class_metrics'][class_name]
        print(f"{class_name:<15} {metrics['total']:<10} {metrics['correct']:<10} "
              f"{metrics['misclassified']:<15} {metrics['accuracy']:>10.2f}% {metrics['recall']:>10.2f}%")
    
    print("\nPer-Class Performance (SuSy 6 Classes - Relevant Classes):")
    print(f"{'SuSy Class':<25} {'Total':<10} {'Correct':<10} {'Misclassified':<15} {'Accuracy':<12} {'Recall':<12}")
    
    for class_name in ['authentic', 'dalle-3-images', 'midjourney-images', 'midjourney_tti']:
        metrics = results['susy_class_metrics'][class_name]
        print(f"{class_name:<25} {metrics['total']:<10} {metrics['correct']:<10} "
              f"{metrics['misclassified']:<15} {metrics['accuracy']:>10.2f}% {metrics['recall']:>10.2f}%")


def save_results(results, output_dir, class_names):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'overall_accuracy': results['overall_accuracy'],
        'class_metrics': results['class_metrics'],
        'confusion_matrix': results['confusion_matrix'],
    }
    
    summary_path = output_dir / 'test_summary_transfer.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    predictions_data = []
    for i in range(len(results['predictions'])):
        pred_class = class_names[results['predictions'][i]]
        true_class = class_names[results['labels'][i]]
        probs = results['probabilities'][i]
        
        predictions_data.append({
            'image_path': results['paths'][i],
            'true_class': true_class,
            'predicted_class': pred_class,
            'correct': pred_class == true_class,
            'confidence': float(probs[results['predictions'][i]]),
            'probabilities': {
                class_names[j]: float(probs[j]) for j in range(len(class_names))
            }
        })
    
    predictions_path = output_dir / 'predictions_transfer.json'
    with open(predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    print(f"Detailed predictions saved to: {predictions_path}")
    
    misclassified = [p for p in predictions_data if not p['correct']]
    if misclassified:
        misclassified_path = output_dir / 'misclassified_transfer.json'
        with open(misclassified_path, 'w') as f:
            json.dump(misclassified, f, indent=2)
        print(f"Misclassified images ({len(misclassified)}) saved to: {misclassified_path}")


def save_original_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'class_metrics': results['class_metrics'],
        'susy_class_metrics': results['susy_class_metrics']
    }
    
    summary_path = output_dir / 'test_summary_original.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Original model summary saved to: {summary_path}")


def analyze_confidence(results, class_names):
    print("\nCONFIDENCE ANALYSIS - TRANSFER LEARNING MODEL")
    
    confidences = []
    correct_confidences = []
    incorrect_confidences = []
    
    for i in range(len(results['predictions'])):
        pred = results['predictions'][i]
        label = results['labels'][i]
        prob = results['probabilities'][i][pred]
        
        confidences.append(prob)
        if pred == label:
            correct_confidences.append(prob)
        else:
            incorrect_confidences.append(prob)
    
    print(f"\nOverall Confidence Statistics:")
    print(f"  Mean: {np.mean(confidences):.3f}")
    print(f"  Median: {np.median(confidences):.3f}")
    print(f"  Std: {np.std(confidences):.3f}")
    print(f"  Min: {np.min(confidences):.3f}")
    print(f"  Max: {np.max(confidences):.3f}")
    
    if correct_confidences:
        print(f"\nCorrect Predictions Confidence:")
        print(f"  Mean: {np.mean(correct_confidences):.3f}")
        print(f"  Median: {np.median(correct_confidences):.3f}")
    
    if incorrect_confidences:
        print(f"\nIncorrect Predictions Confidence:")
        print(f"  Mean: {np.mean(incorrect_confidences):.3f}")
        print(f"  Median: {np.median(incorrect_confidences):.3f}")
    
    print(f"\nPredictions by Confidence Threshold:")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        above_threshold = [i for i, c in enumerate(confidences) if c >= threshold]
        if above_threshold:
            correct_above = sum(1 for i in above_threshold 
                              if results['predictions'][i] == results['labels'][i])
            acc_above = 100 * correct_above / len(above_threshold)
            print(f"  >={threshold:.1f}: {len(above_threshold)} images ({acc_above:.2f}% accuracy)")


def main():
    print("\nDownloading SuSy model from HuggingFace...")
    susy_path = hf_hub_download(repo_id="HPAI-BSC/SuSy", filename="SuSy.pt")

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])

    print(f"\nLoading test data from: {data_dir}")
    test_dataset = TestDataset(data_dir, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    class_names = ['Art', 'dalle2', 'midjourney']

    print("\n" + "="*70)
    print("EVALUATING TRANSFER LEARNING MODEL")
    print("="*70)
    
    print(f"\nLoading transfer learning model from checkpoint: {checkpoint_path}")
    transfer_model = SuSyTransferLearning(susy_path, freeze_backbone=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        transfer_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        transfer_model.load_state_dict(checkpoint)

    transfer_model = transfer_model.to(device)
    transfer_model.eval()
    print("Transfer learning model loaded successfully!")

    transfer_results = evaluate_transfer_model(transfer_model, test_loader, device, class_names)
    print_transfer_results(transfer_results, class_names)
    analyze_confidence(transfer_results, class_names)
    save_results(transfer_results, './test_results', class_names)

    print("\n" + "="*70)
    print("EVALUATING ORIGINAL SUSY MODEL")
    print("="*70)
    
    print("\nLoading original SuSy model...")
    original_model = SuSyOriginal(susy_path)
    original_model = original_model.to(device)
    original_model.eval()
    print("Original SuSy model loaded successfully!")

    original_results = evaluate_original_model(original_model, test_loader, device)
    print_original_results(original_results)
    save_original_results(original_results, './test_results')

    print("\n" + "="*70)
    print("Testing completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()