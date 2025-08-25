"""
学習ユーティリティ：不均衡対策とTTA
"""
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import torchvision.transforms.functional as TF

def make_weighted_sampler(labels):
    """クラス不均衡対策用のWeightedRandomSampler作成"""
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

class FocalLoss(nn.Module):
    """Focal Loss：難例と少数クラスを強調"""
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # e.g., tensor([w0, w1]) or None
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, logits, targets):
        ce = self.ce(logits, targets)  # [N]
        pt = torch.softmax(logits, dim=1)[torch.arange(len(targets)), targets].clamp_min(1e-6)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            w = self.alpha.to(logits.device)[targets]
            focal = w * focal
        return focal.mean() if self.reduction == "mean" else focal.sum()

def predict_with_tta(model, image_tensor, device):
    """Test-Time Augmentation：推論安定化"""
    # image_tensor: [1,3,H,W]
    imgs = [image_tensor]
    imgs.append(TF.hflip(image_tensor))
    imgs.append(TF.rotate(image_tensor, 10))
    imgs.append(TF.rotate(image_tensor, -10))
    # 小スケール変換も追加
    imgs.append(TF.affine(image_tensor, angle=0, translate=[0,0], scale=1.1, shear=0))
    imgs.append(TF.affine(image_tensor, angle=0, translate=[0,0], scale=0.9, shear=0))
    
    with torch.no_grad():
        probs = []
        for x in imgs:
            x = x.to(device)
            out = model(x)
            probs.append(torch.softmax(out, dim=1)[:,1].item())
    return float(sum(probs)/len(probs))

def optimize_per_class_thresholds(y_true, y_prob, disease_names, targets=None):
    """クラス別しきい値最適化：感度ターゲット"""
    if targets is None:
        targets = {
            "AK": {"sens": 0.90},
            "Bowen病": {"sens": 0.90},
            "MM": {"sens": 0.95},
            "BCC": {"sens": 0.80},
            "SK": {"spec": 0.90},  # SKは特異度重視
            "default": {"sens": 0.85}
        }
    
    thresholds = {}
    classes = set(disease_names)
    
    for cls in classes:
        idx = [i for i, d in enumerate(disease_names) if d == cls]
        if len(idx) < 5:  # データ少なすぎ回避
            thresholds[cls] = 0.5
            continue
        
        yt = np.array(y_true)[idx]
        yp = np.array(y_prob)[idx]
        goal = targets.get(cls, targets.get("default", {"sens": 0.85}))
        
        best_t, best_metric = 0.5, 0
        
        for t in np.linspace(0.05, 0.95, 91):
            pred = (yp >= t).astype(int)
            tp = ((yt == 1) & (pred == 1)).sum()
            fn = ((yt == 1) & (pred == 0)).sum()
            tn = ((yt == 0) & (pred == 0)).sum()
            fp = ((yt == 0) & (pred == 1)).sum()
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            if "sens" in goal:
                if sens >= goal["sens"] and spec > best_metric:
                    best_t, best_metric = t, spec
            else:  # spec重視
                if spec >= goal["spec"] and sens > best_metric:
                    best_t, best_metric = t, sens
        
        thresholds[cls] = best_t
    
    return thresholds