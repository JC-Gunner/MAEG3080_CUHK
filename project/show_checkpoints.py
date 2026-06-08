import torch
from pathlib import Path

checkpoints = [
    "checkpoints/smoke_finetune/best.pt",
    "checkpoints/finetune_correct/best.pt", 
    "checkpoints/finetune_improved/best.pt",
    "checkpoints/train_from_best/best.pt"
]

print("\n" + "="*60)
print("CHECKPOINT SUMMARY")
print("="*60)

best_acc = 0
best_ckpt = None

for ckpt_path in checkpoints:
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        val_acc = ckpt.get('val_acc', 0.0)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_ckpt = ckpt_path
        
        print(f"\n📊 {ckpt_path}")
        if 'val_acc' in ckpt:
            print(f"   Accuracy: {ckpt['val_acc']:.4f}")
        if 'epoch' in ckpt:
            print(f"   Epochs: {ckpt['epoch']}")
        if 'model_args' in ckpt:
            args = ckpt['model_args']
            print(f"   Model args: base_width={args.get('base_width', '?')}, blocks={args.get('blocks_per_stage', '?')}")
    except Exception as e:
        print(f"\n❌ {ckpt_path}: {str(e)[:50]}")

print(f"\n" + "="*60)
print(f"🏆 BEST CHECKPOINT: {best_ckpt}")
print(f"   Accuracy: {best_acc:.4f}")
print("="*60 + "\n")
