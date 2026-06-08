import torch
import torch.nn as nn
from pathlib import Path
from model import SEResNet
from utils import build_dataloaders

# Load checkpoint
ckpt_path = Path("checkpoints/smoke_finetune/best.pt")
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"\n=== Checkpoint Info ===")
print(f"Model name: {ckpt.get('model_name', 'unknown')}")
if 'model_args' in ckpt:
    for k, v in ckpt['model_args'].items():
        print(f"  {k}: {v}")
if 'val_acc' in ckpt:
    print(f"Best val accuracy: {ckpt['val_acc']:.4f}")
if 'epoch' in ckpt:
    print(f"Trained for {ckpt['epoch']} epochs")

# Load model
print(f"\n=== Loading Model ===")
model = SEResNet(num_classes=100, base_width=32, blocks_per_stage=(2,2,2,2), use_se=True)
model_dict = ckpt.get('model_state_dict', {})
matched = 0
for name, param in model.state_dict().items():
    if name in model_dict:
        model.state_dict()[name].copy_(model_dict[name])
        matched += 1

print(f"Loaded {matched}/{len(model.state_dict())} weights")
model = model.eval()

# Load data
print(f"\n=== Loading Data ===")
_, eval_loader, _ = build_dataloaders(
    data_dir="data",
    batch_size=64,
    num_workers=0,
    aug="basic"
)

# Evaluate
print(f"\n=== Evaluating ===")
correct = 0
total = 0

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(eval_loader):
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            acc = 100 * correct / total
            print(f"  Batch {batch_idx+1}/{len(eval_loader)} - Accuracy: {acc:.2f}%")

final_acc = 100 * correct / total
print(f"\n✅ Final Val Accuracy: {final_acc:.4f}%")
