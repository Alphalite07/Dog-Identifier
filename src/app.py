import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

# Import your trained model
from model import DogEmbedder

# --- Configuration ---
WEIGHTS_PATH = "../weights/indie_dog_reid_v1.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.85 # Minimum cosine similarity to declare a match

# --- 1. Load Models ---
print("Loading YOLO11 Nano...")
yolo_model = YOLO("yolo11n.pt")

print("Loading Metric Learning Embedder...")
embedder = DogEmbedder(embedding_dim=128).to(DEVICE)
if os.path.exists(WEIGHTS_PATH):
    embedder.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    embedder.eval()
else:
    print(f"Warning: Could not find weights at {WEIGHTS_PATH}. Running with untrained weights!")

# Image preprocessing matching the training phase
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. The Vector Database ---
# In a real scenario, you run your training images through the trained model once,
# average the vectors for each dog, and save them to a file (e.g., dog_db.pt).
def load_vector_database():
    """
    Mock function representing your stored dog identities.
    Returns a dictionary of { "Dog_Name": Tensor(1, 128) }
    """
    # Replace this with torch.load('path_to_your_database.pt')
    return {
        "Tommy (Dog_001)": torch.randn(1, 128).to(DEVICE), 
        "Rani (Dog_002)": torch.randn(1, 128).to(DEVICE)
    }

dog_database = load_vector_database()

# --- 3. The Core Inference Pipeline ---
def identify_dog(input_image):
    if input_image is None:
        return "Please upload an image.", None

    # Convert PIL to OpenCV format for YOLO
    cv_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    
    # Run YOLO Detection
    results = yolo_model(cv_image, verbose=False)[0]
    
    # Find the dog with the highest confidence
    best_crop = None
    max_conf = 0
    for box in results.boxes:
        if int(box.cls[0]) == 16: # COCO ID 16 is 'dog'
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf = conf
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                best_crop = cv_image[ymin:ymax, xmin:xmax]

    if best_crop is None or best_crop.size == 0:
        return "No dog detected in the image.", input_image

    # Convert crop back to PIL for PyTorch transforms
    best_crop_pil = Image.fromarray(cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB))
    input_tensor = transform(best_crop_pil).unsqueeze(0).to(DEVICE)

    # Generate the mathematical embedding
    with torch.no_grad():
        query_embedding = embedder(input_tensor)

    # Compare against the database using Cosine Similarity
    best_match = "Unknown Indie Dog"
    highest_similarity = -1.0

    for dog_name, db_embedding in dog_database.items():
        # Ensure db_embedding is normalized just like the query
        db_embedding = F.normalize(db_embedding, p=2, dim=1)
        sim = F.cosine_similarity(query_embedding, db_embedding).item()
        
        if sim > highest_similarity:
            highest_similarity = sim
            if sim >= CONFIDENCE_THRESHOLD:
                best_match = dog_name

    # Draw a bounding box for the UI output
    output_image = cv_image.copy()
    cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    cv2.putText(output_image, f"{best_match} ({highest_similarity:.2f})", 
                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    return f"Match: {best_match}\nConfidence Score: {highest_similarity:.2f}", output_image

# --- 4. The Gradio Web Interface ---
interface = gr.Interface(
    fn=identify_dog,
    inputs=gr.Image(type="pil", label="Upload Photo of Stray Dog"),
    outputs=[
        gr.Textbox(label="System Output"),
        gr.Image(type="numpy", label="YOLO Detection & Identification")
    ],
    title="🐾 Indian Indie Dog Re-Identification",
    description="Upload an image of an Indie dog. The system uses YOLO11 to isolate the animal and a custom ResNet-18 Metric Learning model to match its biometric signature against known profiles.",
    theme="huggingface",
    allow_flagging="never"
)

if __name__ == "__main__":
    # Runs the web server on localhost:7860
    interface.launch(server_name="0.0.0.0", server_port=7860)