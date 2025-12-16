import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from huggingface_hub import hf_hub_download


class SuSyTransferLearning(nn.Module):
    """
    Applying additional projectio layer to perform transfer learning for SuSy: 6 -> 3 classes
    Classes: Authentic, Midjourney, DALL-E 3
    """

    def __init__(self, susy_repo_id: str = "HPAI-BSC/SuSy", susy_filename: str = "SuSy.pt", freeze_backbone = True):
        super().__init__()

        # Load pretrained SuSy model
        model_path = hf_hub_download(repo_id=susy_repo_id, filename=susy_filename)
        self.susy_model = torch.jit.load(model_path)

        # Freeze weight from pretrained model
        if freeze_backbone:
            for param in self.susy_model.parameters():
                param.requires_grad  = False
            print("SuSy backbone frozen")

        # New projection layer
        self.projection = nn.Linear(6,3)

        # Initialize projection to set up for good grouping
        self._initialize_projection()

    def _initialize_projection(self):
        with torch.no_grad():
            # Row 0: Authentic
            self.projection.weight[0, 0] = 2.0   # authentic -> authentic
            self.projection.weight[0, 1] = -1.0  # dalle-3 -> not authentic
            self.projection.weight[0, 2] = -0.5  # diffusiondb -> not authentic
            self.projection.weight[0, 3] = -1.0  # midjourney-images -> not authentic
            self.projection.weight[0, 4] = -1.0  # midjourney_tti -> not authentic
            self.projection.weight[0, 5] = -0.5  # realisticSDXL -> not authentic

            # Row 1: Midjourney (combine both midjourney classes)
            self.projection.weight[1, 0] = -1.0  # authentic -> not midjourney
            self.projection.weight[1, 1] = -1.0  # dalle-3 -> not midjourney
            self.projection.weight[1, 2] = -0.5  # diffusiondb -> weak signal
            self.projection.weight[1, 3] = 2.0   # midjourney-images -> midjourney
            self.projection.weight[1, 4] = 2.0   # midjourney_tti -> midjourney
            self.projection.weight[1, 5] = -0.5  # realisticSDXL -> weak signal

            # Row 2: DALL-E 3
            self.projection.weight[2, 0] = -1.0  # authentic -> not dalle3
            self.projection.weight[2, 1] = 2.0   # dalle-3 -> dalle3
            self.projection.weight[2, 2] = -0.5  # diffusiondb -> weak signal
            self.projection.weight[2, 3] = -1.0  # midjourney-images -> not dalle3
            self.projection.weight[2, 4] = -1.0  # midjourney_tti -> not dalle3
            self.projection.weight[2, 5] = -0.5  # realisticSDXL -> weak signal

            # Initialize bias to zero
            self.projection.bias.zero_()

    def forward(self, x):
        # Get the 6 class logits
        with torch.set_grad_enabled(self.training and self._requires_grad_susy()):
            susy_logits = self.susy_model(x)


        # Project to 3 classes
        output = self.projection(susy_logits)

        return output

    def _requires_grad_susy(self):
        """Check if SuSy backbone requires gradients"""
        return any(p.requires_grad for p in self.susy_model.parameters())

    def unfreeze_backbone(self):
        """Unfreeze the SuSy backbone for fine-tuning"""
        for param in self.susy_model.parameters():
            param.requires_grad = True
        print("SuSy backbone unfrozen for fine-tuning")


class ImageDataset(Dataset):
    """
    Dataset for 3-class image clf
    """
    def __init__(self, data_dir, split = 'train', transform = None) -> None:
        self.data_dir = Path(data_dir) / split
        self.transform = transform

        # Class mapping
        self.class_to_idx = {
            'authentic': 0,
            'midjourney': 1,
            'dalle3': 2
        }

        # Load all image paths
        self.samples = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_dir / class_name

            # Class data dir dne
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist!")
                continue

            # Collect all images
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), class_idx))

            if len(self.samples) == 0:
                raise ValueError(f"No images found in {self.data_dir}")

            print(f"Loaded {len(self.samples)} images from {split} set")
            self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class distribution"""
        class_counts = {name: 0 for name in self.class_to_idx.keys()}
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        for _, label in self.samples:
            class_counts[idx_to_class[label]] += 1

        print(f"  Class distribution: {class_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    """Get data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Bc SuSy trained on normalized resnet data
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # For recall calculation
    true_positives = [0, 0, 0]  # Per class
    false_negatives = [0, 0, 0]  # Per class

    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Calculate recall metrics
        for i in range(len(labels)):
            true_label = labels[i].item()
            pred_label = predicted[i].item()

            if pred_label == true_label:
                true_positives[true_label] += 1
            else:
                false_negatives[true_label] += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    # Calculate per-class recall
    class_recalls = {}
    class_names = ['authentic', 'midjourney', 'dalle3']
    for i, name in enumerate(class_names):
        total_class = true_positives[i] + false_negatives[i]
        recall = 100 * true_positives[i] / total_class if total_class > 0 else 0
        class_recalls[name] = recall

    # Average recall across classes
    avg_recall = sum(class_recalls.values()) / len(class_recalls)

    return epoch_loss, epoch_acc, avg_recall, class_recalls


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Per-class metrics
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    # For recall
    true_positives = [0, 0, 0]
    false_negatives = [0, 0, 0]

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy and recall
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()

                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
                    true_positives[label] += 1
                else:
                    false_negatives[label] += 1

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    # Calculate per-class accuracy and recall
    class_names = ['authentic', 'midjourney', 'dalle3']
    class_acc = {}
    class_recalls = {}

    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc[name] = 100 * class_correct[i] / class_total[i]
            class_recalls[name] = 100 * true_positives[i] / class_total[i]
        else:
            class_acc[name] = 0.0
            class_recalls[name] = 0.0

    # Average recall
    avg_recall = sum(class_recalls.values()) / len(class_recalls)

    return epoch_loss, epoch_acc, avg_recall, class_recalls, class_acc

def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    output_dir,
    num_epochs_stage1=10,
    num_epochs_stage2=10,
    initial_lr=0.001,
    stage2_lr=0.0001,
):
    """
    Two-stage training pipeline:
    Stage 1: Train projection layer only (backbone frozen)
    Stage 2: Fine-tune entire model (backbone unfrozen)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'stage1': {
            'train_loss': [],
            'train_acc': [],
            'train_recall': [],
            'val_loss': [],
            'val_acc': [],
            'val_recall': []
        },
        'stage2': {
            'train_loss': [],
            'train_acc': [],
            'train_recall': [],
            'val_loss': [],
            'val_acc': [],
            'val_recall': []
        },
    }

    best_val_acc = 0.0


    # STAGE 1: Train projection layer only
    print("STAGE 1: Training Projection Layer (Backbone Frozen)")

    # Only optimize projection layer parameters
    optimizer = Adam(model.projection.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    for epoch in range(num_epochs_stage1):
        print(f"\n[Stage 1] Epoch {epoch+1}/{num_epochs_stage1}")

        # Train
        train_loss, train_acc, train_recall, train_class_recalls = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_recall, val_class_recalls, class_acc = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step(val_acc)
        print(f"Updated lr = {scheduler.get_last_lr()}")

        # Store history
        history['stage1']['train_loss'].append(train_loss)
        history['stage1']['train_acc'].append(train_acc)
        history['stage1']['train_recall'].append(train_recall)
        history['stage1']['val_loss'].append(val_loss)
        history['stage1']['val_acc'].append(val_acc)
        history['stage1']['val_recall'].append(val_recall)

        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Recall: {train_recall:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Recall: {val_recall:.2f}%")
        print(f"Per-class Val Recall: {val_class_recalls}")
        print(f"Per-class Val Acc: {class_acc}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / 'best_model_stage1.pth'
            save_checkpoint(model, optimizer, epoch, {
                'train_acc': train_acc,
                'train_recall': train_recall,
                'val_acc': val_acc,
                'val_recall': val_recall,
                'class_acc': class_acc,
                'class_recalls': val_class_recalls
            }, checkpoint_path)

    # STAGE 2: Fine-tune entire model
    print("STAGE 2: Fine-tuning Entire Model (Backbone Unfrozen)")

    # Unfreeze backbone
    model.unfreeze_backbone()

    # Optimize all parameters with lower learning rate
    optimizer = Adam(model.parameters(), lr=stage2_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    for epoch in range(num_epochs_stage2):
        print(f"\n[Stage 2] Epoch {epoch+1}/{num_epochs_stage2}")

        # Train
        train_loss, train_acc, train_recall, train_class_recalls = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_recall, val_class_recalls, class_acc = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step(val_acc)
        print(f"Updated lr = {scheduler.get_last_lr()}")

        # Store history
        history['stage2']['train_loss'].append(train_loss)
        history['stage2']['train_acc'].append(train_acc)
        history['stage2']['train_recall'].append(train_recall)
        history['stage2']['val_loss'].append(val_loss)
        history['stage2']['val_acc'].append(val_acc)
        history['stage2']['val_recall'].append(val_recall)

        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Recall: {train_recall:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Recall: {val_recall:.2f}%")
        print(f"Per-class Val Recall: {val_class_recalls}")
        print(f"Per-class Val Acc: {class_acc}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / 'best_model_stage2.pth'
            save_checkpoint(model, optimizer, epoch, {
                'train_acc': train_acc,
                'train_recall': train_recall,
                'val_acc': val_acc,
                'val_recall': val_recall,
                'class_acc': class_acc,
                'class_recalls': val_class_recalls
            }, checkpoint_path)

    # Save final model and history
    final_model_path = output_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    print("\n" + "="*70)
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*70)

    return model, history

def main():

    # Set parameters 
    data_dir = '/content/drive/MyDrive/data'
    output_dir = '/content/drive/MyDrive/models'
    batch_size = 32
    epochs_stage1 = 10
    epochs_stage2 = 10
    lr_stage1 = 0.001
    lr_stage2 = 0.0001
    num_workers = 2  #

    # Set device (auto-detect)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Download SuSy model from HuggingFace
    print("\nDownloading SuSy model from HuggingFace...")
    susy_path = hf_hub_download(repo_id="HPAI-BSC/SuSy", filename="SuSy.pt")
    print(f"SuSy model downloaded to: {susy_path}")

    # Create model
    print("\nCreating transfer learning model...")
    model = SuSyTransferLearning(freeze_backbone=True)
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (Stage 1): {trainable_params:,}")

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ImageDataset(data_dir, split='train', transform=train_transform)
    val_dataset = ImageDataset(data_dir, split='val', transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Train model
    print("\nStarting training...")
    print(f"Stage 1: {epochs_stage1} epochs (lr={lr_stage1})")
    print(f"Stage 2: {epochs_stage2} epochs (lr={lr_stage2})")

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        num_epochs_stage1=epochs_stage1,
        num_epochs_stage2=epochs_stage2,
        initial_lr=lr_stage1,
        stage2_lr=lr_stage2,
    )

    print("\nTraining completed successfully!")
    print(f"Models saved in: {output_dir}")

