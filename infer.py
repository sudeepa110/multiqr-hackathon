import os
import json
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def run_inference_stage1(model_path, input_dir, output_file, save_images=False):
    """Run YOLO detection only (Stage 1) on images."""

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"❌ Input directory not found: {input_dir}")

    # Ensure output dirs
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    save_images_dir = "src/models/predict_stage1_annotated"
    if save_images:
        os.makedirs(save_images_dir, exist_ok=True)

    # Load YOLO model
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)

    # Get images
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise RuntimeError(f"❌ No images found in {input_dir}")

    print(f"🚀 Running Stage 1 on {len(image_files)} images...")
    submission = []

    for img_file in tqdm(image_files, desc="Stage 1"):
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
            qrs.append({"bbox": [float(x) for x in box]})

            if save_images:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, "qr_code", (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        submission.append({"image_id": img_file, "qrs": qrs})

        if save_images:
            save_path = os.path.join(save_images_dir, img_file)
            cv2.imwrite(save_path, img)

    with open(output_file, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"✅ Stage 1 submission saved: {output_file}")
    if save_images:
        print(f"✅ Annotated images saved in: {save_images_dir}")


if __name__ == "__main__":
    MODEL_PATH = r"D:\multiqr-hackaton\src\models\qr_detector\weights\best.pt"
    INPUT_DIR = r"D:\multiqr-hackaton\src\datasets\QR_DS\test_images"
    OUTPUT_DIR = r"D:\multiqr-hackaton\outputs"
    SAVE_IMAGES = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_inference_stage1(
        model_path=MODEL_PATH,
        input_dir=INPUT_DIR,
        output_file=os.path.join(OUTPUT_DIR, "submission_detection_1.json"),
        save_images=SAVE_IMAGES
    )

    print("\n🎉 Stage 1 detection submission is ready!")
