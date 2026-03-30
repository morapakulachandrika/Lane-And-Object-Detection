# ==========================================
# ⚠️ MUST BE FIRST: OpenMP Conflict Fix
# ==========================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP error on Windows
os.environ["OMP_NUM_THREADS"] = "1"          # Optional: limit threads

# ==========================================
# Now import other libraries
# ==========================================
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO

# ==========================================
# MAIN EXECUTION BLOCK (REQUIRED FOR WINDOWS)
# ==========================================
if __name__ == '__main__':
    
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    DATASET_ROOT = r"D:\ObjectDetection\bdd100k"
    DATASET_YAML = os.path.join(DATASET_ROOT, "dataset.yaml")  # or "dataset.yaml"
    OUTPUT_PROJECT = r"D:\ObjectDetection\Training_Results"
    MODEL_NAME = "yolov5s.pt"
    EPOCHS = 100
    IMAGE_SIZE = 640
    BATCH_SIZE = 16  # Reduce to 8 if you get CUDA OOM errors
    DEVICE = 0       # 0 for GPU, 'cpu' for CPU
    WORKERS = 0      # ⚠️ Set to 0 on Windows to avoid multiprocessing issues

    # ==========================================
    # 2. GPU AVAILABILITY CHECK
    # ==========================================
    print("="*60)
    print("SYSTEM CHECK")
    print("="*60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ CUDA Available: Yes")
        print(f"🚀 GPU Device: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.2f} GB")
        print(f"⚙️  Using Device: cuda:{DEVICE}")
    else:
        print("❌ CUDA Available: No - Training will run on CPU")
        DEVICE = 'cpu'

    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(f"❌ Dataset YAML not found at {DATASET_YAML}.\n"
                                f"Please ensure BDD100K is converted to YOLO format.")

    # ==========================================
    # 3. INITIALIZE MODEL
    # ==========================================
    print(f"\n📥 Loading model: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    # ==========================================
    # 4. TRAIN THE MODEL
    # ==========================================
    print("\n🔥 Starting Training...")
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        amp=True,               # Automatic Mixed Precision (faster on GPU)
        workers=WORKERS,        # ⚠️ CRITICAL: Set to 0 on Windows to fix multiprocessing error
        project=OUTPUT_PROJECT,
        name="YOLOv5_BDD100K_Run",
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True              # Generates results.png automatically
    )

    # ==========================================
    # 5. PRINT METRICS
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING COMPLETED. FINAL METRICS:")
    print("="*60)

    TRAINING_FOLDER = os.path.join(OUTPUT_PROJECT, "YOLOv5_BDD100K_Run")
    RESULTS_CSV = os.path.join(TRAINING_FOLDER, "results.csv")
    RESULTS_PLOT = os.path.join(TRAINING_FOLDER, "results.png")

    if os.path.exists(RESULTS_CSV):
        try:
            df = pd.read_csv(RESULTS_CSV)
            last_row = df.iloc[-1]
            cols = df.columns
            
            def get_metric(keywords):
                for c in cols:
                    if all(k in c for k in keywords):
                        return last_row[c]
                return "N/A"

            precision = get_metric(['precision'])
            recall = get_metric(['recall'])
            map50 = get_metric(['mAP50'])
            map95 = get_metric(['mAP50-95'])

            print(f"🎯 Precision:    {precision}")
            print(f"🔍 Recall:       {recall}")
            print(f"📈 mAP@0.5:      {map50}")
            print(f"📈 mAP@0.5-0.95: {map95}")
            print(f"💾 Results Saved: {TRAINING_FOLDER}")
            
        except Exception as e:
            print(f"⚠️ Could not parse results.csv: {e}")
    else:
        print(f"❌ Results CSV not found at {RESULTS_CSV}")

    # ==========================================
    # 6. DISPLAY ACCURACY GRAPH
    # ==========================================
    if os.path.exists(RESULTS_PLOT):
        print("\n📊 Displaying Training Results Graph...")
        try:
            img = mpimg.imread(RESULTS_PLOT)
            plt.figure(figsize=(20, 15))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"YOLOv5 Training Metrics\nSaved in: {TRAINING_FOLDER}")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not display graph: {e}")
    else:
        print(f"❌ Results Plot not found at {RESULTS_PLOT}")

    print("\n✅ Done!")
