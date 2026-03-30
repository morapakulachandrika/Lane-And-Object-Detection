# ============================================
# INTEGRATED LANE + OBJECT DETECTION SYSTEM
# ENet-SAD (Lane) + YOLOv5s (Objects) - CPU Optimized
# WITH REAL-TIME ACCURACY & CONFIDENCE METRICS
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
import time
import traceback

# Force light theme for maximum visibility on all systems
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# ============================================
# CONFIGURATION
# ============================================

PROJECT_DIR = 'D:/Project_Final_Year'
if not os.path.exists(PROJECT_DIR):
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"⚠️  Using script directory as project folder: {PROJECT_DIR}")

LANE_MODEL_PATH = os.path.join(PROJECT_DIR, 'enet_sad_best.pth')
YOLO_MODEL_PATH = os.path.join(PROJECT_DIR, 'yolov5s.pt')

# Model input size (H, W) - matches your training
RESIZE_SHAPE = (288, 800)

# Device selection (CPU for your system)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"Lane Model: {LANE_MODEL_PATH}")
print(f"YOLO Model: {YOLO_MODEL_PATH}\n")

# ============================================
# ENET-SAD MODEL ARCHITECTURE (EXACT MATCH TO TRAINING)
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
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1), padding=(padding, 0), dilation=dilation, bias=False),
                nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size), padding=(0, padding), dilation=dilation, bias=False)
            )
        else:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
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
        return self.prelu3(out)

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
        return self.prelu3(out)

class BottleneckUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0, relu=True):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = activation()
        self.deconv = nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
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
        return self.prelu3(out)

class ENet_SAD(nn.Module):
    def __init__(self, num_classes=2, encoder_relu=False, decoder_relu=True, sad=True):
        super().__init__()
        self.sad = sad
        self.initial = InitialBlock(3, 13)
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
# MODEL LOADING WITH AUTO-RECOVERY
# ============================================

def load_models():
    print("="*70)
    print("LOADING MODELS")
    print("="*70)
    
    # Load ENet-SAD
    print("\n1. Loading ENet-SAD Lane Detection Model...")
    if not os.path.exists(LANE_MODEL_PATH):
        error_msg = f"ENet-SAD weights not found at:\n{LANE_MODEL_PATH}\n\nPlease train the model first using your training script!"
        print(f"\n❌ {error_msg}")
        messagebox.showerror("Model Error", error_msg)
        sys.exit(1)
    
    try:
        lane_model = ENet_SAD(num_classes=2, sad=True).to(DEVICE)
        lane_model.load_state_dict(torch.load(LANE_MODEL_PATH, map_location=DEVICE))
        lane_model.eval()
        params = sum(p.numel() for p in lane_model.parameters())
        print(f"   ✓ Successfully loaded ({params/1e6:.2f}M parameters)")
    except Exception as e:
        print(f"\n❌ Failed to load ENet-SAD: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # Load YOLOv5s with auto-download
    print("\n2. Loading YOLOv5s Object Detection Model...")
    try:
        from ultralytics import YOLO as UltralyticsYOLO
        
        if not os.path.exists(YOLO_MODEL_PATH):
            print("   ⚠️  Weights not found - downloading YOLOv5s (14MB)...")
            yolo_model = UltralyticsYOLO('yolov5s')
            os.makedirs(os.path.dirname(YOLO_MODEL_PATH), exist_ok=True)
            yolo_model.model.save(YOLO_MODEL_PATH)
            print(f"   ✓ Saved to: {YOLO_MODEL_PATH}")
        else:
            yolo_model = UltralyticsYOLO(YOLO_MODEL_PATH)
        
        print("   ✓ Loaded successfully via Ultralytics")
        return lane_model, yolo_model
    
    except Exception as e:
        print(f"\n❌ Failed to load YOLOv5s: {str(e)}")
        print("\nSOLUTIONS:")
        print("  1. Install ultralytics: pip install ultralytics")
        print("  2. Or download weights manually from:")
        print("     https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt")
        traceback.print_exc()
        sys.exit(1)

# Load models BEFORE GUI
print("\n" + "="*70)
print("INITIALIZING SYSTEM")
print("="*70)
lane_model, yolo_model = load_models()
print("\n✅ BOTH MODELS LOADED SUCCESSFULLY")
print(f"   • Lane Detection: ENet-SAD ({'GPU' if DEVICE.type=='cuda' else 'CPU'})")
print(f"   • Object Detection: YOLOv5s (Ultralytics)")
print("="*70)

# ============================================
# DETECTION FUNCTIONS WITH ACCURACY METRICS
# ============================================

def detect_lanes_with_metrics(img):
    """Detect lanes and return mask + quantitative metrics"""
    h_orig, w_orig = img.shape[:2]
    
    # Preprocess (identical to training)
    img_resized = cv2.resize(img, (RESIZE_SHAPE[1], RESIZE_SHAPE[0]))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        output = lane_model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        conf_map = probs[0, 1].cpu().numpy()
    
    # Resize to original resolution
    pred_full = cv2.resize(pred.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    conf_full = cv2.resize(conf_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    # Apply confidence threshold (0.5 = your training threshold)
    lane_mask = np.where(conf_full > 0.5, pred_full, 0).astype(np.uint8)
    
    # ROI: Focus on drivable area (bottom 60%)
    roi_start = int(h_orig * 0.4)
    lane_mask[:roi_start, :] = 0
    
    # Calculate metrics
    total_pixels = h_orig * w_orig
    lane_pixels = np.sum(lane_mask > 0)
    coverage_pct = (lane_pixels / total_pixels) * 100
    
    # Quality assessment based on coverage
    if coverage_pct > 5.0:
        quality = "Excellent"
        quality_color = (0, 255, 0)  # Green
    elif coverage_pct > 2.0:
        quality = "Good"
        quality_color = (0, 255, 255)  # Yellow
    elif coverage_pct > 0.5:
        quality = "Fair"
        quality_color = (0, 165, 255)  # Orange
    else:
        quality = "Poor"
        quality_color = (0, 0, 255)  # Red
    
    # Average confidence over detected lanes
    avg_conf = conf_full[lane_mask > 0].mean() if lane_pixels > 0 else 0.0
    
    metrics = {
        'mask': (lane_mask * 255).astype(np.uint8),
        'coverage_pct': coverage_pct,
        'avg_confidence': avg_conf,
        'quality': quality,
        'quality_color': quality_color,
        'lane_pixels': lane_pixels
    }
    
    return metrics

def detect_objects_with_metrics(img):
    """Detect objects and return results + quantitative metrics"""
    # Ultralytics inference
    results = yolo_model.predict(
        img, 
        conf=0.4,    # Optimized for CPU reliability
        iou=0.45,
        device=DEVICE,
        verbose=False
    )[0]
    
    # Calculate metrics
    boxes = results.boxes
    num_detections = len(boxes)
    
    if num_detections > 0:
        confidences = [float(box.conf[0]) for box in boxes]
        avg_conf = np.mean(confidences)
        max_conf = np.max(confidences)
        
        # Count vehicles vs vulnerable road users
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        vehicles = sum(1 for box in boxes if results.names[int(box.cls[0])] in vehicle_classes)
        vulnerable = num_detections - vehicles
        
        # Quality assessment
        if avg_conf > 0.75:
            quality = "Excellent"
            quality_color = (0, 255, 0)
        elif avg_conf > 0.6:
            quality = "Good"
            quality_color = (0, 255, 255)
        elif avg_conf > 0.45:
            quality = "Fair"
            quality_color = (0, 165, 255)
        else:
            quality = "Poor"
            quality_color = (0, 0, 255)
    else:
        avg_conf = 0.0
        max_conf = 0.0
        vehicles = 0
        vulnerable = 0
        quality = "None"
        quality_color = (128, 128, 128)
    
    metrics = {
        'results': results,
        'num_detections': num_detections,
        'avg_confidence': avg_conf,
        'max_confidence': max_conf,
        'vehicles': vehicles,
        'vulnerable': vulnerable,
        'quality': quality,
        'quality_color': quality_color
    }
    
    return metrics

def detect_all_with_accuracy(img):
    """Full pipeline with accuracy/confidence visualization"""
    result = img.copy()
    h, w = img.shape[:2]
    
    # 1. Lane detection with metrics
    lane_metrics = detect_lanes_with_metrics(img)
    lane_mask = lane_metrics['mask']
    
    # Apply lane overlay if detected
    if lane_metrics['lane_pixels'] > 0:
        overlay = np.zeros_like(result)
        overlay[lane_mask > 127] = [0, 255, 0]  # Green lanes
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
    
    # 2. Object detection with metrics
    obj_metrics = detect_objects_with_metrics(img)
    results = obj_metrics['results']
    
    # Draw object detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = results.names[cls]
        
        # Color by class type
        if class_name in ['car', 'truck', 'bus', 'motorcycle']:
            color = (0, 0, 255)  # Red for vehicles
        elif class_name in ['person', 'bicycle']:
            color = (255, 0, 0)  # Blue for vulnerable
        else:
            color = (255, 255, 0)  # Yellow for others
        
        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f'{class_name} {conf:.2f}'
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 3. Add ACCURACY & CONFIDENCE PANEL (critical feature)
    panel_h = 110
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:, :] = (30, 35, 45)  # Dark blue-gray background
    
    # Lane detection metrics
    lane_color = lane_metrics['quality_color']
    cv2.putText(panel, "LANE DETECTION", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    lane_text = f"Coverage: {lane_metrics['coverage_pct']:.2f}% | Confidence: {lane_metrics['avg_confidence']:.3f} | Quality: {lane_metrics['quality']}"
    cv2.putText(panel, lane_text, (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, lane_color, 2)
    
    # Object detection metrics
    obj_color = obj_metrics['quality_color']
    cv2.putText(panel, "OBJECT DETECTION", (20, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    obj_text = f"Detections: {obj_metrics['num_detections']} ({obj_metrics['vehicles']} vehicles, {obj_metrics['vulnerable']} vulnerable) | Avg Conf: {obj_metrics['avg_confidence']:.3f} | Quality: {obj_metrics['quality']}"
    cv2.putText(panel, obj_text, (20, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, obj_color, 2)
    
    # Add system info
    sys_info = f"ENet-SAD + YOLOv5s | Device: {'GPU' if DEVICE.type == 'cuda' else 'CPU'}"
    cv2.putText(panel, sys_info, (w - 550, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    
    # Combine panel with result
    result = np.vstack([result, panel])
    
    # Return full result + metrics for logging
    return result, {
        'lane_coverage': lane_metrics['coverage_pct'],
        'lane_confidence': lane_metrics['avg_confidence'],
        'lane_quality': lane_metrics['quality'],
        'object_count': obj_metrics['num_detections'],
        'object_confidence': obj_metrics['avg_confidence'],
        'object_quality': obj_metrics['quality']
    }

def create_gui():
    root = tk.Tk()
    root.title("🚗 Integrated Lane & Object Detection System")
    
    # Force visibility
    root.attributes('-topmost', True)
    root.update()
    root.attributes('-topmost', False)
    
    # Light theme
    root.configure(bg='white')
    
    # Set minimum size to ensure buttons are always visible
    root.minsize(1100, 750)
    
    # Center window
    window_width, window_height = 1300, 850
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')
    
    # Header
    header = tk.Frame(root, bg='#2c3e50', height=80)
    header.pack(fill='x', side='top')
    tk.Label(header, text="🚗 Integrated Lane & Object Detection System", 
            font=('Arial', 22, 'bold'), bg='#2c3e50', fg='white', pady=15).pack()
    tk.Label(header, text="ENet-SAD (Lane Detection) + YOLOv5s (Object Detection) | Real-time Accuracy Metrics", 
            font=('Arial', 11), bg='#2c3e50', fg='#1abc9c').pack()
    
    # Status bar
    status = tk.Label(root, text="✅ System Ready - Load an image to start detection", 
                     font=('Arial', 13, 'bold'), bg='#e8f4f8', fg='#27ae60', pady=10)
    status.pack(fill='x', side='top')
    
    # Content area (will be packed LAST after reserving bottom space)
    content = tk.Frame(root, bg='white', padx=20, pady=15)
    
    # Image display frame
    img_frame = tk.Frame(content, bg='#f8f9fa', relief='groove', borderwidth=2)
    img_frame.pack(fill='both', expand=True, pady=10)
    
    # Original image panel
    left_panel = tk.Frame(img_frame, bg='white', padx=15, pady=10)
    left_panel.pack(side='left', fill='both', expand=True)
    tk.Label(left_panel, text="Original Image", font=('Arial', 14, 'bold'), 
            bg='white', fg='#2c3e50').pack(pady=(0, 10))
    orig_canvas = tk.Label(left_panel, bg='#e9ecef', width=60, height=30,
                          text="Click 'Load Image' to start", 
                          fg='#6c757d', font=('Arial', 12, 'italic'))
    orig_canvas.pack(fill='both', expand=True)
    
    # Result image panel
    right_panel = tk.Frame(img_frame, bg='white', padx=15, pady=10)
    right_panel.pack(side='right', fill='both', expand=True)
    tk.Label(right_panel, text="Detection Result with Accuracy Metrics", 
            font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50').pack(pady=(0, 10))
    result_canvas = tk.Label(right_panel, bg='#e9ecef', width=60, height=30,
                           text="Process image to see lane coverage % and object confidence", 
                           fg='#6c757d', font=('Arial', 12, 'italic'))
    result_canvas.pack(fill='both', expand=True)
    
    # BUTTON FRAME (packed BEFORE content to reserve space)
    btn_frame = tk.Frame(root, bg='white', pady=15)
    
    btn_style = {
        'font': ('Arial', 13, 'bold'),
        'width': 18, 
        'height': 2, 
        'bd': 2,
        'relief': 'raised',
        'cursor': 'hand2',
        'padx': 10,
        'pady': 5
    }
    
    load_btn = tk.Button(btn_frame, text="📁 LOAD IMAGE", 
                        bg='#3498db', fg='white', activebackground='#2980b9', 
                        activeforeground='white', **btn_style)
    
    detect_btn = tk.Button(btn_frame, text="🔍 DETECT ALL", 
                          bg='#27ae60', fg='white', activebackground='#229954',
                          activeforeground='white', state='disabled', **btn_style)
    
    save_btn = tk.Button(btn_frame, text="💾 SAVE RESULT", 
                        bg='#e67e22', fg='white', activebackground='#d35400',
                        activeforeground='white', state='disabled', **btn_style)
    
    load_btn.pack(side='left', padx=15)
    detect_btn.pack(side='left', padx=15)
    save_btn.pack(side='left', padx=15)
    
    # Footer (packed FIRST to reserve bottom space)
    footer = tk.Frame(root, bg='#ecf0f1', pady=12)
    tk.Label(footer, text="💡 Accuracy Metrics: Lane Coverage % (higher = better) | Object Confidence (0.0-1.0, higher = more certain)", 
            font=('Arial', 10), bg='#ecf0f1', fg='#7f8c8d').pack()
    
    # ===== CRITICAL PACKING ORDER (BOTTOM-UP) =====
    footer.pack(side='bottom', fill='x')      # 1. Footer at very bottom
    btn_frame.pack(side='bottom', fill='x')   # 2. Buttons above footer
    content.pack(side='top', fill='both', expand=True)  # 3. Content fills middle
    # ===============================================
    
    # Store state
    root.current_img = [None]
    root.result_img = [None]
    root.detection_metrics = [None]
    
    # ===== CORRECTED BUTTON CALLBACKS (NO SCOPE ISSUES) =====
    def choose_image():
        # CORRECT: Define path variable BEFORE using it
        file_path = filedialog.askopenfilename(
            title="Select Road Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        
        # Early return if dialog cancelled
        if not file_path:
            return
        
        try:
            # USE file_path HERE (not undefined 'path')
            img = cv2.imread(file_path)
            if img is None or img.size == 0:
                raise ValueError("Invalid or corrupted image file")
            
            root.current_img[0] = img
            
            # Resize for display only (preserve full resolution for processing)
            h, w = img.shape[:2]
            scale = min(550 / w, 350 / h)  # Reduced height for better fit
            disp_w, disp_h = int(w * scale), int(h * scale)
            disp_img = cv2.resize(img, (disp_w, disp_h))
            
            # Convert to RGB for Tkinter
            disp_rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(disp_rgb))
            
            orig_canvas.configure(image=photo, text='')
            orig_canvas.image = photo
            
            # Reset result panel
            result_canvas.configure(image='', 
                                  text="Click 'Detect All' to run lane + object detection\nwith real-time accuracy metrics", 
                                  fg='#6c757d', font=('Arial', 11, 'italic'))
            result_canvas.image = None
            
            # Enable detection button
            detect_btn.config(state='normal')
            save_btn.config(state='disabled')
            
            # Update status
            status.config(text=f"✅ Loaded: {os.path.basename(file_path)} ({w}x{h})", fg='#27ae60')
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image:\n{str(e)}")
            status.config(text="❌ Image load failed", fg='#e74c3c')
            print(f"Image load error: {e}")
            traceback.print_exc()
    
    def run_detection():
        if root.current_img[0] is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        # Update UI immediately
        status.config(text="⏳ Processing... (Lane Detection + Object Detection)", fg='#e67e22')
        root.update()
        
        try:
            start_time = time.time()
            result_img, metrics = detect_all_with_accuracy(root.current_img[0])
            processing_time = time.time() - start_time
            
            root.result_img[0] = result_img
            root.detection_metrics[0] = metrics
            
            # Resize for display (preserve aspect ratio)
            h, w = result_img.shape[:2]
            scale = min(550 / w, 350 / h)  # Reduced height
            disp_w, disp_h = int(w * scale), int(h * scale)
            disp_img = cv2.resize(result_img, (disp_w, disp_h))
            
            # Convert to RGB for Tkinter
            disp_rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(disp_rgb))
            
            result_canvas.configure(image=photo, text='')
            result_canvas.image = photo
            
            # Enable save button
            save_btn.config(state='normal')
            
            # Update status with performance metrics
            fps = 1.0 / processing_time if processing_time > 0 else 0
            status_text = (f"✅ Detection Complete! | "
                          f"Lane Coverage: {metrics['lane_coverage']:.2f}% | "
                          f"Objects: {metrics['object_count']} | "
                          f"Time: {processing_time*1000:.1f}ms ({fps:.1f} FPS)")
            status.config(text=status_text, fg='#27ae60')
            
            # Print metrics to console for logging
            print("\n" + "="*70)
            print("DETECTION ACCURACY METRICS")
            print("="*70)
            print(f"Lane Detection:")
            print(f"  • Coverage:       {metrics['lane_coverage']:.2f}%")
            print(f"  • Avg Confidence: {metrics['lane_confidence']:.3f}")
            print(f"  • Quality:        {metrics['lane_quality']}")
            print(f"\nObject Detection:")
            print(f"  • Detections:     {metrics['object_count']}")
            print(f"  • Avg Confidence: {metrics['object_confidence']:.3f}")
            print(f"  • Quality:        {metrics['object_quality']}")
            print(f"\nPerformance:")
            print(f"  • Processing Time: {processing_time*1000:.1f} ms")
            print(f"  • Estimated FPS:   {fps:.1f}")
            print("="*70)
            
        except Exception as e:
            messagebox.showerror("Detection Error", f"Detection failed:\n{str(e)}")
            status.config(text="❌ Detection failed - check console for details", fg='#e74c3c')
            print(f"\nDETECTION ERROR: {e}")
            traceback.print_exc()
    
    def save_result():
        if root.result_img[0] is None:
            messagebox.showwarning("Warning", "No detection result to save!")
            return
        
        default_name = f"detection_result_{int(time.time())}.jpg"
        save_path = filedialog.asksaveasfilename(
            title="Save Detection Result",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        
        if not save_path:  # User cancelled save dialog
            return
        
        try:
            cv2.imwrite(save_path, root.result_img[0])
            
            # Save metrics to text file
            metrics_path = os.path.splitext(save_path)[0] + "_metrics.txt"
            with open(metrics_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write("DETECTION ACCURACY METRICS\n")
                f.write("="*70 + "\n")
                f.write(f"Image: {os.path.basename(save_path)}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("LANE DETECTION:\n")
                f.write(f"  Coverage:       {root.detection_metrics[0]['lane_coverage']:.2f}%\n")
                f.write(f"  Avg Confidence: {root.detection_metrics[0]['lane_confidence']:.3f}\n")
                f.write(f"  Quality:        {root.detection_metrics[0]['lane_quality']}\n\n")
                f.write("OBJECT DETECTION:\n")
                f.write(f"  Detections:     {root.detection_metrics[0]['object_count']}\n")
                f.write(f"  Avg Confidence: {root.detection_metrics[0]['object_confidence']:.3f}\n")
                f.write(f"  Quality:        {root.detection_metrics[0]['object_quality']}\n\n")
                f.write("PERFORMANCE:\n")
                f.write(f"  Device:         {DEVICE}\n")
                f.write(f"  Model:          ENet-SAD + YOLOv5s\n")
                f.write("="*70 + "\n")
            
            messagebox.showinfo("Success", 
                              f"Result saved to:\n{save_path}\n\n"
                              f"Metrics saved to:\n{metrics_path}")
            status.config(text=f"💾 Saved: {os.path.basename(save_path)} + metrics", fg='#27ae60')
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image:\n{str(e)}")
            status.config(text="❌ Save failed", fg='#e74c3c')
            print(f"Save error: {e}")
            traceback.print_exc()
    # ========================================================
    
    # Bind button commands
    load_btn.config(command=choose_image)
    detect_btn.config(command=run_detection)
    save_btn.config(command=save_result)
    
    # Force window to front
    root.lift()
    root.focus_force()
    root.after(100, lambda: root.focus_force())
    
    print("\n" + "="*70)
    print("✅ SYSTEM READY - BUTTONS VISIBLE AT BOTTOM")
    print("="*70)
    print("USAGE:")
    print("  1. Click BLUE 'LOAD IMAGE' button → Select road image")
    print("  2. Click GREEN 'DETECT ALL' button → See results with accuracy metrics")
    print("  3. Click ORANGE 'SAVE RESULT' button → Save image + metrics report")
    print("\nTROUBLESHOOTING:")
    print("  • If buttons aren't visible: Drag window bottom edge DOWN to resize taller")
    print("  • Minimum window size enforced: 1100×750 pixels")
    print("="*70 + "\n")
    
    return root

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    try:
        root = create_gui()
        root.mainloop()
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: {e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")
