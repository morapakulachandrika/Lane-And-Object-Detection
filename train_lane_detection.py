# ============================================
# COMPLETE ENET-SAD LANE DETECTION TRAINING
# For Python IDLE with TuSimple Dataset
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For saving plots in IDLE
import matplotlib.pyplot as plt
from glob import glob
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time

print("="*70)
print("ENET-SAD LANE DETECTION TRAINING")
print("="*70)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# ============================================
# ENET-SAD MODEL ARCHITECTURE
# ============================================

class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=13):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels + in_channels)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        main = self.conv(x)
        side = self.maxpool(x)
        x = torch.cat([main, side], dim=1)
        x = self.bn(x)
        return self.prelu(x)

class BottleneckRegular(nn.Module):
    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=1, 
                 dilation=1, asymmetric=False, dropout_prob=0, relu=False):
        super().__init__()
        
        internal_channels = channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
            
        self.conv1 = nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = activation()
        
        if asymmetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1), 
                         padding=(padding, 0), dilation=dilation, bias=False),
                nn.BatchNorm2d(internal_channels),
                activation(),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size), 
                         padding=(0, padding), dilation=dilation, bias=False)
            )
        else:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, 
                                   padding=padding, dilation=dilation, bias=False)
        
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = activation()
        
        self.conv3 = nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.prelu3 = activation()
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out += identity
        out = self.prelu3(out)
        return out

class BottleneckDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0, relu=False):
        super().__init__()
        
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = activation()
        
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = activation()
        
        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu3 = activation()
        
    def forward(self, x):
        main = self.conv1(x)
        main = self.bn1(main)
        main = self.prelu1(main)
        main = self.conv2(main)
        main = self.bn2(main)
        main = self.prelu2(main)
        main = self.conv3(main)
        main = self.bn3(main)
        main = self.dropout(main)
        
        side = self.maxpool(x)
        diff_channels = main.size(1) - side.size(1)
        if diff_channels > 0:
            side = F.pad(side, (0, 0, 0, 0, 0, diff_channels))
        
        out = main + side
        out = self.prelu3(out)
        return out

class BottleneckUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0, relu=True):
        super().__init__()
        
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = activation()
        
        self.deconv = nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=3, 
                                         stride=2, padding=1, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = activation()
        
        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        self.side_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.side_bn = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.prelu3 = activation()
        
    def forward(self, x):
        main = self.conv1(x)
        main = self.bn1(main)
        main = self.prelu1(main)
        main = self.deconv(main)
        main = self.bn2(main)
        main = self.prelu2(main)
        main = self.conv3(main)
        main = self.bn3(main)
        main = self.dropout(main)
        
        side = self.side_conv(x)
        side = self.side_bn(side)
        side = self.upsample(side)
        
        out = main + side
        out = self.prelu3(out)
        return out

class ENet_SAD(nn.Module):
    def __init__(self, num_classes=2, encoder_relu=False, decoder_relu=True, sad=True):
        super().__init__()
        
        self.sad = sad
        self.initial = InitialBlock(3, 13)
        
        # Encoder
        self.downsample1_0 = BottleneckDownsample(16, 64, relu=encoder_relu, dropout_prob=0.01)
        self.regular1_1 = BottleneckRegular(64, relu=encoder_relu, dropout_prob=0.01)
        self.regular1_2 = BottleneckRegular(64, relu=encoder_relu, dropout_prob=0.01)
        self.regular1_3 = BottleneckRegular(64, relu=encoder_relu, dropout_prob=0.01)
        self.regular1_4 = BottleneckRegular(64, relu=encoder_relu, dropout_prob=0.01)
        
        self.downsample2_0 = BottleneckDownsample(64, 128, relu=encoder_relu, dropout_prob=0.1)
        self.regular2_1 = BottleneckRegular(128, relu=encoder_relu, dropout_prob=0.1)
        self.dilated2_2 = BottleneckRegular(128, dilation=2, padding=2, relu=encoder_relu, dropout_prob=0.1)
        self.asymmetric2_3 = BottleneckRegular(128, kernel_size=5, padding=2, asymmetric=True, relu=encoder_relu, dropout_prob=0.1)
        self.dilated2_4 = BottleneckRegular(128, dilation=4, padding=4, relu=encoder_relu, dropout_prob=0.1)
        self.regular2_5 = BottleneckRegular(128, relu=encoder_relu, dropout_prob=0.1)
        self.dilated2_6 = BottleneckRegular(128, dilation=8, padding=8, relu=encoder_relu, dropout_prob=0.1)
        self.asymmetric2_7 = BottleneckRegular(128, kernel_size=5, asymmetric=True, padding=2, relu=encoder_relu, dropout_prob=0.1)
        self.dilated2_8 = BottleneckRegular(128, dilation=16, padding=16, relu=encoder_relu, dropout_prob=0.1)
        
        self.regular3_0 = BottleneckRegular(128, relu=encoder_relu, dropout_prob=0.1)
        self.dilated3_1 = BottleneckRegular(128, dilation=2, padding=2, relu=encoder_relu, dropout_prob=0.1)
        self.asymmetric3_2 = BottleneckRegular(128, kernel_size=5, padding=2, asymmetric=True, relu=encoder_relu, dropout_prob=0.1)
        self.dilated3_3 = BottleneckRegular(128, dilation=4, padding=4, relu=encoder_relu, dropout_prob=0.1)
        self.regular3_4 = BottleneckRegular(128, relu=encoder_relu, dropout_prob=0.1)
        self.dilated3_5 = BottleneckRegular(128, dilation=8, padding=8, relu=encoder_relu, dropout_prob=0.1)
        self.asymmetric3_6 = BottleneckRegular(128, kernel_size=5, asymmetric=True, padding=2, relu=encoder_relu, dropout_prob=0.1)
        self.dilated3_7 = BottleneckRegular(128, dilation=16, padding=16, relu=encoder_relu, dropout_prob=0.1)
        
        # Decoder
        self.upsample4_0 = BottleneckUpsample(128, 64, relu=decoder_relu, dropout_prob=0.1)
        self.regular4_1 = BottleneckRegular(64, relu=decoder_relu, dropout_prob=0.1)
        self.regular4_2 = BottleneckRegular(64, relu=decoder_relu, dropout_prob=0.1)
        
        self.upsample5_0 = BottleneckUpsample(64, 16, relu=decoder_relu, dropout_prob=0.1)
        self.regular5_1 = BottleneckRegular(16, relu=decoder_relu, dropout_prob=0.1)
        
        self.deconv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, bias=False)
        
        if self.sad:
            self.attention2 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1), nn.Sigmoid())
            self.attention3 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1), nn.Sigmoid())
        
    def forward(self, x):
        x = self.initial(x)
        x = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        
        x = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        
        if self.sad:
            att2 = self.attention2(x)
        
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)
        
        if self.sad:
            att3 = self.attention3(x)
        
        x = self.upsample4_0(x)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        
        x = self.upsample5_0(x)
        x = self.regular5_1(x)
        x = self.deconv(x)
        
        if self.sad and self.training:
            return x, att2, att3
        return x

# ============================================
# DATASET CLASS
# ============================================

class LaneDataset(Dataset):
    def __init__(self, data_pairs, img_size=(288, 800)):
        self.data_pairs = data_pairs
        self.img_size = img_size
        
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.data_pairs[idx]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img = img.astype(np.float32) / 255.0
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        
        return img, mask

# ============================================
# LOAD DATASET
# ============================================

print('\n' + '='*70)
print('LOADING DATASET')
print('='*70)

image_folder = "D:/Project_Final_Year/images"
mask_folder = "D:/Project_Final_Year/mask"

image_files = sorted(glob(os.path.join(image_folder, '*.jpg')))
mask_files = sorted(glob(os.path.join(mask_folder, '*.png')))

print(f'Found {len(image_files)} images')
print(f'Found {len(mask_files)} masks')

# Pair images with masks
paired_data = []
for img_path in image_files:
    img_name = os.path.basename(img_path)
    mask_name = img_name.replace('.jpg', '.png')
    mask_path = os.path.join(mask_folder, mask_name)
    
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.max() > 0:
            paired_data.append((img_path, mask_path))

print(f'Valid image-mask pairs: {len(paired_data)}')

# Train/val split
train_data, val_data = train_test_split(paired_data, test_size=0.2, random_state=42)

train_dataset = LaneDataset(train_data)
val_dataset = LaneDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

print(f'Training samples: {len(train_dataset)}')
print(f'Validation samples: {len(val_dataset)}')

# ============================================
# CALCULATE CLASS WEIGHTS
# ============================================

print('\nCalculating class weights...')
class_counts = torch.zeros(2)
for _, masks in tqdm(train_loader, desc='Analyzing data'):
    class_counts[0] += (masks == 0).sum()
    class_counts[1] += (masks == 1).sum()

total = class_counts.sum()
class_weights = total / (2 * class_counts)
class_weights = class_weights.to(device)

print(f'Class weights: Background={class_weights[0]:.2f}, Lane={class_weights[1]:.2f}')

# ============================================
# INITIALIZE MODEL
# ============================================

print('\n' + '='*70)
print('INITIALIZING MODEL')
print('='*70)

model = ENet_SAD(num_classes=2, sad=True).to(device)
params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {params/1e6:.2f}M')

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# ============================================
# METRIC CALCULATION
# ============================================

def calculate_metrics(outputs, masks):
    preds = torch.argmax(outputs, dim=1)
    
    # Pixel accuracy
    correct = (preds == masks).float().sum()
    total = masks.numel()
    pixel_acc = (correct / total).item()
    
    # IoU for each class
    ious = []
    precisions = []
    recalls = []
    f1s = []
    
    for cls in range(2):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)
        
        tp = (pred_cls & mask_cls).float().sum().item()
        fp = (pred_cls & ~mask_cls).float().sum().item()
        fn = (~pred_cls & mask_cls).float().sum().item()
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    return {
        'pixel_accuracy': pixel_acc,
        'bg_iou': ious[0],
        'lane_iou': ious[1],
        'lane_precision': precisions[1],
        'lane_recall': recalls[1],
        'lane_f1': f1s[1]
    }

# ============================================
# TRAINING FUNCTIONS
# ============================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    metrics_sum = {'pixel_accuracy': 0, 'lane_iou': 0, 'lane_precision': 0, 'lane_recall': 0, 'lane_f1': 0}
    count = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        if model.sad:
            outputs, att2, att3 = model(images)
            loss = criterion(outputs, masks) + 0.1 * F.mse_loss(att2, att3.detach())
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        metrics = calculate_metrics(outputs, masks)
        
        total_loss += loss.item()
        for key in metrics_sum:
            metrics_sum[key] += metrics[key]
        count += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{metrics["pixel_accuracy"]*100:.1f}%',
            'iou': f'{metrics["lane_iou"]:.3f}'
        })
    
    avg_metrics = {k: v/count for k, v in metrics_sum.items()}
    avg_metrics['loss'] = total_loss / count
    return avg_metrics

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics_sum = {'pixel_accuracy': 0, 'bg_iou': 0, 'lane_iou': 0, 'lane_precision': 0, 'lane_recall': 0, 'lane_f1': 0}
    count = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            metrics = calculate_metrics(outputs, masks)
            
            total_loss += loss.item()
            for key in metrics_sum:
                metrics_sum[key] += metrics[key]
            count += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{metrics["pixel_accuracy"]*100:.1f}%'
            })
    
    avg_metrics = {k: v/count for k, v in metrics_sum.items()}
    avg_metrics['loss'] = total_loss / count
    return avg_metrics

# ============================================
# TRAINING LOOP
# ============================================

num_epochs = 30
best_iou = 0

history = {
    'train_loss': [], 'train_acc': [], 'train_iou': [],
    'val_loss': [], 'val_acc': [], 'val_iou': [],
    'val_precision': [], 'val_recall': [], 'val_f1': []
}

print('\n' + '='*70)
print('STARTING TRAINING')
print('='*70 + '\n')

start_time = time.time()

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 70)
    
    # Train
    train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics = validate(model, val_loader, criterion, device)
    scheduler.step()
    
    # Store history
    history['train_loss'].append(train_metrics['loss'])
    history['train_acc'].append(train_metrics['pixel_accuracy'])
    history['train_iou'].append(train_metrics['lane_iou'])
    history['val_loss'].append(val_metrics['loss'])
    history['val_acc'].append(val_metrics['pixel_accuracy'])
    history['val_iou'].append(val_metrics['lane_iou'])
    history['val_precision'].append(val_metrics['lane_precision'])
    history['val_recall'].append(val_metrics['lane_recall'])
    history['val_f1'].append(val_metrics['lane_f1'])
    
    # Print metrics
    print(f'\n📊 Training:')
    print(f'  Loss: {train_metrics["loss"]:.4f} | Acc: {train_metrics["pixel_accuracy"]*100:.2f}% | IoU: {train_metrics["lane_iou"]:.4f}')
    
    print(f'\n📊 Validation:')
    print(f'  Loss: {val_metrics["loss"]:.4f} | Acc: {val_metrics["pixel_accuracy"]*100:.2f}%')
    print(f'  BG IoU: {val_metrics["bg_iou"]:.4f} | Lane IoU: {val_metrics["lane_iou"]:.4f}')
    print(f'  Precision: {val_metrics["lane_precision"]:.4f} | Recall: {val_metrics["lane_recall"]:.4f} | F1: {val_metrics["lane_f1"]:.4f}')
    
    # Save best model
    if val_metrics['lane_iou'] > best_iou:
        best_iou = val_metrics['lane_iou']
        torch.save(model.state_dict(), 'D:/Project_Final_Year/enet_sad_best.pth')
        print(f'\n✅ Model saved! Best IoU: {best_iou:.4f}')

training_time = time.time() - start_time

print('\n' + '='*70)
print('✅ TRAINING COMPLETE!')
print('='*70)
print(f'Training time: {training_time/60:.2f} minutes')
print(f'Best validation IoU: {best_iou:.4f}')
print('='*70)

# ============================================
# PLOT TRAINING HISTORY
# ============================================

print('\nGenerating training plots...')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot([x*100 for x in history['train_acc']], label='Train', linewidth=2)
axes[0, 1].plot([x*100 for x in history['val_acc']], label='Validation', linewidth=2)
axes[0, 1].set_title('Pixel Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# IoU
axes[0, 2].plot(history['train_iou'], label='Train', linewidth=2)
axes[0, 2].plot(history['val_iou'], label='Validation', linewidth=2)
axes[0, 2].set_title('Lane IoU', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('IoU')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history['val_precision'], linewidth=2, color='green')
axes[1, 0].set_title('Validation Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history['val_recall'], linewidth=2, color='orange')
axes[1, 1].set_title('Validation Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].grid(True, alpha=0.3)

# F1 Score
axes[1, 2].plot(history['val_f1'], linewidth=2, color='purple')
axes[1, 2].set_title('Validation F1-Score', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('F1-Score')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/Project_Final_Year/training_history.png', dpi=150, bbox_inches='tight')
print('✓ Saved: training_history.png')

# ============================================
# FINAL SUMMARY REPORT
# ============================================

report = f"""
{'='*70}
ENET-SAD TRAINING REPORT
{'='*70}

Dataset:
  Training samples:   {len(train_dataset)}
  Validation samples: {len(val_dataset)}
  Total samples:      {len(paired_data)}

Model:
  Architecture:  ENet-SAD
  Parameters:    {params/1e6:.2f}M
  Device:        {device}

Training:
  Epochs:        {num_epochs}
  Batch size:    4
  Learning rate: 1e-3
  Time:          {training_time/60:.2f} minutes

Final Metrics:
  Training Loss:        {history['train_loss'][-1]:.4f}
  Training Accuracy:    {history['train_acc'][-1]*100:.2f}%
  Training IoU:         {history['train_iou'][-1]:.4f}
  
  Validation Loss:      {history['val_loss'][-1]:.4f}
  Validation Accuracy:  {history['val_acc'][-1]*100:.2f}%
  Validation Lane IoU:  {history['val_iou'][-1]:.4f}
  Validation Precision: {history['val_precision'][-1]:.4f}
  Validation Recall:    {history['val_recall'][-1]:.4f}
  Validation F1-Score:  {history['val_f1'][-1]:.4f}
  
  Best Validation IoU:  {best_iou:.4f}

Saved Files:
  • D:/Project_Final_Year/enet_sad_best.pth
  • D:/Project_Final_Year/training_history.png
  • D:/Project_Final_Year/training_report.txt

{'='*70}
"""

with open('D:/Project_Final_Year/training_report.txt', 'w') as f:
    f.write(report)

print(report)
print('✓ Saved: training_report.txt')

print('\n✅ All done! Check the output folder for saved files.')
