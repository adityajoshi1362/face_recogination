"""
=============================================================
  CNN-Based Facial Recognition System  (v2 – DeconvSkip Head)
  Uses: FaceNet (InceptionResnetV1) + MTCNN + OpenCV
  Modification: Adds a small deconv+skip refinement block on
                top of the frozen FaceNet features before the
                SVM classifier.  Trains fast, same pipeline.
  Saves model as: model2.pkl
=============================================================
"""

import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
FACES_DIR = "known_faces_folder"               # Root folder containing subfolders per person
MODEL_SAVE_PATH = "model2.pkl"                 # << Changed to model2.pkl
CONFIDENCE_THRESHOLD = 0.55                    # Below this → "Unknown"
IMAGE_SIZE = (160, 160)                        # FaceNet input size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {DEVICE}")


# ─────────────────────────────────────────────
#  DECONV-SKIP REFINEMENT HEAD
#  Takes 512-d FaceNet embedding, reshapes it
#  into a small feature map, applies deconv +
#  skip connection, then projects back to 512-d.
# ─────────────────────────────────────────────
class DeconvSkipHead(nn.Module):
    """
    Lightweight refinement head with deconv layers and skip connections.
    
    Flow:
        512 → reshape(32, 4, 4) → DeconvBlock(32→64, 8×8) → DeconvBlock(64→32, 16×16)
         ↓ (skip from 4×4 via upsample)                              ↓
        concat skip → Conv1×1 merge → AdaptiveAvgPool → Linear → 512
    """

    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim

        # Project 512 → 32*4*4 = 512 (reshape-friendly)
        self.fc_in = nn.Linear(embed_dim, 32 * 4 * 4)
        self.bn_in = nn.BatchNorm1d(32 * 4 * 4)

        # Deconv block 1: 32ch 4×4 → 64ch 8×8
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Deconv block 2: 64ch 8×8 → 32ch 16×16
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Skip: upsample 32ch 4×4 → 32ch 16×16 and merge
        self.skip_up = nn.Upsample(size=(16, 16), mode="bilinear", align_corners=False)

        # After concat skip (32 + 32 = 64 channels) → merge to 32
        self.merge = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Final pooling + projection back to 512
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_out = nn.Linear(32, embed_dim)

        # Residual gate (learnable scaling of the refinement)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """x: (B, 512) raw FaceNet embedding."""
        identity = x  # skip connection at embedding level

        h = self.fc_in(x)
        h = self.bn_in(h)
        h = torch.relu(h)
        h = h.view(-1, 32, 4, 4)  # (B, 32, 4, 4)

        skip = h  # save for spatial skip

        h = self.deconv1(h)   # (B, 64, 8, 8)
        h = self.deconv2(h)   # (B, 32, 16, 16)

        # Spatial skip connection: upsample original 4×4 → 16×16 and concat
        skip_up = self.skip_up(skip)  # (B, 32, 16, 16)
        h = torch.cat([h, skip_up], dim=1)  # (B, 64, 16, 16)
        h = self.merge(h)  # (B, 32, 16, 16)

        h = self.pool(h).flatten(1)  # (B, 32)
        h = self.fc_out(h)           # (B, 512)

        # Gated residual: output = identity + gate * refined
        out = identity + self.gate * h
        return out


# ─────────────────────────────────────────────
#  INITIALIZE MODELS
# ─────────────────────────────────────────────
def load_models():
    """Load MTCNN face detector, FaceNet backbone, and DeconvSkip head."""
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        keep_all=False,
        device=DEVICE
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    # Our lightweight refinement head
    head = DeconvSkipHead(embed_dim=512).eval().to(DEVICE)
    print("[INFO] DeconvSkipHead loaded (lightweight deconv + skip refinement)")

    return mtcnn, resnet, head


# ─────────────────────────────────────────────
#  EXTRACT FACE EMBEDDING FROM IMAGE
# ─────────────────────────────────────────────
def get_embedding(mtcnn, resnet, head, img_rgb: np.ndarray):
    """
    Given an RGB image (numpy array), detect face and return 512-d
    refined embedding (FaceNet → DeconvSkipHead).
    Returns None if no face detected.
    """
    pil_img = Image.fromarray(img_rgb)
    face_tensor = mtcnn(pil_img)   # Returns cropped & normalized face tensor or None

    if face_tensor is None:
        return None

    # Add batch dim, move to device
    face_tensor = face_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        raw_emb = resnet(face_tensor)    # (1, 512) from FaceNet
        refined_emb = head(raw_emb)      # (1, 512) after deconv+skip refinement

    return refined_emb.squeeze().cpu().numpy()


# ─────────────────────────────────────────────
#  TRAINING PHASE
# ─────────────────────────────────────────────
def train(faces_dir: str, mtcnn, resnet, head):
    """
    Scan faces_dir, extract refined embeddings for all images,
    train an SVM classifier, save to disk as model2.pkl.
    """
    embeddings = []
    labels = []
    failed = 0

    persons = [
        p for p in os.listdir(faces_dir)
        if os.path.isdir(os.path.join(faces_dir, p))
    ]

    if not persons:
        raise ValueError(f"No subfolders found in '{faces_dir}'.")

    print(f"\n[TRAIN] Found {len(persons)} person(s): {persons}")

    for person in persons:
        person_dir = os.path.join(faces_dir, person)
        img_files = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        print(f"  → {person}: {len(img_files)} image(s)")

        for img_file in img_files:
            img_path = os.path.join(person_dir, img_file)
            img_bgr = cv2.imread(img_path)

            if img_bgr is None:
                print(f"    [WARN] Could not read: {img_path}")
                failed += 1
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            emb = get_embedding(mtcnn, resnet, head, img_rgb)

            if emb is None:
                print(f"    [WARN] No face detected: {img_file}")
                failed += 1
                continue

            embeddings.append(emb)
            labels.append(person)

    if len(embeddings) == 0:
        raise RuntimeError("No valid face embeddings extracted. Check your images.")

    print(f"\n[TRAIN] Extracted {len(embeddings)} refined embeddings | Failed/Skipped: {failed}")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = np.array(embeddings)

    # Train/test split for evaluation
    if len(set(y)) > 1 and len(X) > 4:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # SVM classifier (probability=True for confidence scores)
    clf = SVC(
        kernel="linear",
        probability=True,
        C=1.0
    )
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    print("\n[TRAIN] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model  →  model2.pkl
    model_data = {"clf": clf, "le": le}
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model_data, f)

    print(f"[TRAIN] Model saved → {MODEL_SAVE_PATH}")
    return clf, le


# ─────────────────────────────────────────────
#  LOAD SAVED MODEL
# ─────────────────────────────────────────────
def load_classifier():
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model '{MODEL_SAVE_PATH}' not found. Run training first."
        )
    with open(MODEL_SAVE_PATH, "rb") as f:
        data = pickle.load(f)
    return data["clf"], data["le"]


# ─────────────────────────────────────────────
#  PREDICT A SINGLE FACE REGION
# ─────────────────────────────────────────────
def predict_face(emb: np.ndarray, clf: SVC, le: LabelEncoder):
    """
    Returns (name, confidence_percent) or ("Unknown", confidence_percent).
    """
    probs = clf.predict_proba([emb])[0]
    max_prob = np.max(probs)
    pred_idx = np.argmax(probs)
    name = le.inverse_transform([pred_idx])[0]

    if max_prob < CONFIDENCE_THRESHOLD:
        return "Unknown", round(max_prob * 100, 1)

    return name, round(max_prob * 100, 1)


# ─────────────────────────────────────────────
#  RECOGNIZE ON A SINGLE IMAGE FILE
# ─────────────────────────────────────────────
def recognize_image(image_path: str, mtcnn, resnet, head, clf, le):
    """
    Run recognition on a static image and show result with OpenCV.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Detect all faces + bounding boxes
    boxes, probs = mtcnn.detect(pil_img)

    if boxes is None:
        print("[INFO] No faces detected in image.")
        cv2.imshow("Result", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)

        face_rgb = img_rgb[y1:y2, x1:x2]
        if face_rgb.size == 0:
            continue

        # Resize to FaceNet input
        face_pil = Image.fromarray(face_rgb).resize(IMAGE_SIZE)
        emb = get_embedding(mtcnn, resnet, head, np.array(face_pil))

        if emb is None:
            label = "Unknown"
            confidence = 0.0
        else:
            label, confidence = predict_face(emb, clf, le)

        # Draw bounding box
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        text = f"{label} ({confidence}%)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img_bgr, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img_bgr, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        print(f"  → Detected: {label} | Confidence: {confidence}%")

    cv2.imshow("Face Recognition Result (v2 DeconvSkip)", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  LIVE WEBCAM RECOGNITION
# ─────────────────────────────────────────────
def recognize_webcam(mtcnn, resnet, head, clf, le):
    """
    Real-time face recognition from webcam.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[LIVE] Starting webcam. Press 'q' to quit.")
    frame_skip = 2   # Process every Nth frame for speed
    frame_count = 0
    last_results = []  # Cache last predictions

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        if frame_count % frame_skip == 0:
            boxes, _ = mtcnn.detect(pil_img)
            last_results = []

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    face_rgb = img_rgb[y1:y2, x1:x2]
                    if face_rgb.size == 0:
                        continue

                    face_pil = Image.fromarray(face_rgb).resize(IMAGE_SIZE)
                    emb = get_embedding(mtcnn, resnet, head, np.array(face_pil))

                    if emb is None:
                        last_results.append((x1, y1, x2, y2, "Unknown", 0.0))
                    else:
                        label, confidence = predict_face(emb, clf, le)
                        last_results.append((x1, y1, x2, y2, label, confidence))

        # Draw cached results every frame (smooth display)
        for (x1, y1, x2, y2, label, confidence) in last_results:
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({confidence}%)"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Live Face Recognition v2  [q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CNN Facial Recognition System (v2 – DeconvSkip Head)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "image", "webcam", "train+webcam", "train+image"],
        default="train+webcam",
        help=(
            "train         → Train model from faces/ folder\n"
            "image         → Recognize face in a static image\n"
            "webcam        → Live webcam recognition\n"
            "train+webcam  → Train then launch webcam (default)\n"
            "train+image   → Train then test on an image"
        )
    )
    parser.add_argument("--image", type=str, help="Path to image for 'image' mode")
    args = parser.parse_args()

    # Load CNN backbone + deconv head
    print("[INIT] Loading FaceNet + MTCNN + DeconvSkipHead...")
    mtcnn, resnet, head = load_models()
    print("[INIT] Models loaded.\n")

    if args.mode in ("train", "train+webcam", "train+image"):
        clf, le = train(FACES_DIR, mtcnn, resnet, head)
    else:
        clf, le = load_classifier()
        print(f"[INFO] Loaded classifier from {MODEL_SAVE_PATH}. Known persons: {list(le.classes_)}")

    if args.mode in ("image", "train+image"):
        img_path = args.image
        if not img_path:
            img_path = input("\nEnter path to test image: ").strip()
        recognize_image(img_path, mtcnn, resnet, head, clf, le)

    elif args.mode in ("webcam", "train+webcam"):
        recognize_webcam(mtcnn, resnet, head, clf, le)


if __name__ == "__main__":
    main()
