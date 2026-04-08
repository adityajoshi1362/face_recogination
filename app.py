import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import k2  # Import the new architecture we created (contains torch import, must be before fastapi)
import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io

app = FastAPI(title="Facial Recognition Backend")

# Global state
models_loaded = False
mtcnn = None
resnet = None
head = None
clf = None
le = None

@app.on_event("startup")
def startup_event():
    global mtcnn, resnet, head, clf, le, models_loaded
    try:
        print("[INFO] Loading FaceNet + MTCNN + DeconvSkipHead...")
        mtcnn, resnet, head = k2.load_models()
        clf, le = k2.load_classifier()
        models_loaded = True
        print("[INFO] Models loaded successfully! Ready for inference.")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")

@app.get("/api/info")
def get_info():
    """Returns backend readiness and known classes"""
    if not models_loaded:
        return {"status": "starting", "message": "Models are loading..."}
    return {
        "status": "ready",
        "classes": list(le.classes_) if le else []
    }

def process_image(img_rgb: np.ndarray, threshold: float):
    """Core logic to extract faces and get predictions."""
    if img_rgb is None or img_rgb.size == 0:
        return {"error": "Invalid image format."}
    
    pil_img = Image.fromarray(img_rgb)
    boxes, _ = mtcnn.detect(pil_img)
    
    if boxes is None:
        return {"faces": [], "original_size": [img_rgb.shape[1], img_rgb.shape[0]]}
    
    results = []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        # Clamp bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
        
        face_rgb = img_rgb[y1:y2, x1:x2]
        if face_rgb.size == 0:
            continue
            
        face_pil = Image.fromarray(face_rgb).resize(k2.IMAGE_SIZE)
        emb = k2.get_embedding(mtcnn, resnet, head, np.array(face_pil))
        
        label = "Unknown"
        confidence = 0.0
        
        if emb is not None:
            raw_label, raw_confidence = k2.predict_face(emb, clf, le)
            # Re-apply caller threshold override
            if raw_confidence >= threshold:
                label = raw_label
            confidence = raw_confidence
            
        results.append({
            "box": [x1, y1, x2, y2],
            "label": label,
            "confidence": confidence
        })
        
    return {
        "faces": results, 
        "original_size": [img_rgb.shape[1], img_rgb.shape[0]]
    }

@app.post("/api/recognize_image")
async def recognize_image(file: UploadFile = File(...), threshold: float = Form(55.0)):
    if not models_loaded:
        return JSONResponse(status_code=503, content={"error": "Models are still initializing"})
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Could not decode file as image."})
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return process_image(img_rgb, threshold)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

class FrameRequest(BaseModel):
    image_base64: str
    threshold: float = 55.0

@app.post("/api/recognize_frame")
def recognize_frame(req: FrameRequest):
    if not models_loaded:
        return JSONResponse(status_code=503, content={"error": "Models are still initializing"})
        
    try:
        # Expected format: "data:image/jpeg;base64,/9j/4AAQSkZJ..."
        if "," in req.image_base64:
            _, encoded = req.image_base64.split(",", 1)
        else:
            encoded = req.image_base64
            
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Invalid base64 frame."})
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return process_image(img_rgb, req.threshold)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Mount frontend files at root
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
