import os
from ultralytics import YOLO

def train_model():
    """
    Train YOLOv8 on QR code dataset with fixed settings.
    """

    # ✅ Fixed dataset config (update if paths change)
    DATA_YAML = "src/datasets/data.yaml"

    # ✅ Fixed training parameters
    EPOCHS = 100
    BATCH_SIZE = 16
    IMG_SIZE = 640
    PATIENCE = 20

    # ✅ Model and output
    PRETRAINED_MODEL = "yolov8n.pt"   # can be 'yolov8s.pt', etc.
    MODEL_NAME = "qr_detector"
    PROJECT_NAME = "src/models"

    # Check dataset config exists
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"❌ data.yaml not found at {DATA_YAML}")

    print(f"✅ Using dataset config: {DATA_YAML}")

    # Load YOLOv8 model
    model = YOLO(PRETRAINED_MODEL)

    print(f"🚀 Starting training with {PRETRAINED_MODEL}...")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        name=MODEL_NAME,
        project=PROJECT_NAME,
        exist_ok=True
    )

    print(f"\n🎉 Training finished!")
    print(f"📂 Model saved in: {os.path.join(PROJECT_NAME, MODEL_NAME)}")


if __name__ == "__main__":
    train_model()
