# -----------------------------
# infer_stage2.py - Stage 2: YOLO Detection + QR Decoding + Type Classification (OpenCV)
# -----------------------------

import os
import json
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# -----------------------------
# QR Type Classification Rules
# -----------------------------
def classify_qr(value):
    """
    Classify QR code type based on its value.
    """
    if value.startswith(("B", "5a0S")):
        return "batch"
    elif value.startswith("MFR"):
        return "manufacturer"
    elif value.startswith("D"):
        return "distributor"
    elif value.startswith("R"):
        return "regulator"
    else:
        return "unknown"

# -----------------------------
# Stage 2 Inference
# -----------------------------
def run_stage2(model_path, input_dir, output_file, save_images=False):
    """
    Run YOLO detection + QR decoding + type classification (Stage 2) using OpenCV QRCodeDetector.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"❌ Input directory not found: {input_dir}")

    # Output directories
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_images_dir = "src/models/predict_stage2_annoatated"
    if save_images:
        os.makedirs(save_images_dir, exist_ok=True)

    # Load YOLO model
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)

    # OpenCV QR detector
    qr_detector = cv2.QRCodeDetector()

    # Load images
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise RuntimeError(f"❌ No images found in {input_dir}")

    print(f"🚀 Running Stage 2 on {len(image_files)} images...")
    submission = []

    for img_file in tqdm(image_files, desc="Stage 2"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping unreadable image {img_file}")
            continue

        results = model.predict(img_path, save=False, conf=0.25, verbose=False)
        r = results[0]
        qrs = []

        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]

            # Decode QR using OpenCV
            qr_value, _, _ = qr_detector.detectAndDecode(crop)
            if not qr_value:
                qr_value = "unknown"

            qr_type = classify_qr(qr_value)

            qrs.append({
                "bbox": [float(x) for x in box],
                "value": qr_value,
                "type": qr_type
            })

            # Draw boxes and QR value/type on image
            if save_images:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{qr_value} ({qr_type})", (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        submission.append({"image_id": img_file, "qrs": qrs})

        if save_images:
            save_path = os.path.join(save_images_dir, img_file)
            cv2.imwrite(save_path, img)

    # Save JSON output
    with open(output_file, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"✅ Stage 2 submission saved: {output_file}")
    if save_images:
        print(f"✅ Annotated images saved in: {save_images_dir}")


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    MODEL_PATH = r"D:\multiqr-hackaton\src\models\qr_detector\weights\best.pt"
    INPUT_DIR = r"D:\multiqr-hackaton\src\datasets\QR_DS\test_images"
    OUTPUT_DIR = r"D:\multiqr-hackaton\outputs"
    SAVE_IMAGES = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_stage2(
        model_path=MODEL_PATH,
        input_dir=INPUT_DIR,
        output_file=os.path.join(OUTPUT_DIR, "submission_detection_2.json"),
        save_images=SAVE_IMAGES
    )

    print("\n🎉 Stage 2 submission is ready!")
