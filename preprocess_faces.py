import cv2
import mediapipe as mp
from pathlib import Path

mp_face = mp.solutions.face_detection


def extract_faces(
    input_dir,
    output_dir,
    img_size=128,
    margin=0.25,
    save_no_face=True,
):
    detector = mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    total_images = 0
    total_faces = 0
    skipped = 0

    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue

        out_class_dir = output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.glob("*"):
            total_images += 1

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            faces_found = False

            if results.detections:
                for i, det in enumerate(results.detections):
                    bbox = det.location_data.relative_bounding_box

                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

                    # ✅ Add margin safely
                    x1 = int(max(0, x - bw * margin))
                    y1 = int(max(0, y - bh * margin))
                    x2 = int(min(w, x + bw * (1 + margin)))
                    y2 = int(min(h, y + bh * (1 + margin)))

                    face = img[y1:y2, x1:x2]

                    # ❌ Skip invalid crops
                    if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
                        continue

                    face = cv2.resize(face, (img_size, img_size))

                    out_path = out_class_dir / f"{img_path.stem}_face{i}.jpg"
                    cv2.imwrite(str(out_path), face)

                    total_faces += 1
                    faces_found = True

            # ✅ Handle no face detected
            if not faces_found:
                if save_no_face:
                    # fallback: center crop
                    min_dim = min(h, w)
                    start_x = (w - min_dim) // 2
                    start_y = (h - min_dim) // 2
                    crop = img[start_y:start_y+min_dim, start_x:start_x+min_dim]

                    crop = cv2.resize(crop, (img_size, img_size))
                    out_path = out_class_dir / f"{img_path.stem}_noface.jpg"
                    cv2.imwrite(str(out_path), crop)
                else:
                    skipped += 1

    print("\n===== PREPROCESS SUMMARY =====")
    print(f"Total images processed: {total_images}")
    print(f"Total faces extracted: {total_faces}")
    print(f"Skipped images: {skipped}")

if __name__ == "__main__":
    extract_faces("facesData/train", "facesData_faces/train")
    extract_faces("facesData/test", "facesData_faces/test")