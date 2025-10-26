import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch
import sys
from detector.crnn_model import CRNNModel
from torchvision import transforms
from django.conf import settings

# Use GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize YOLO text detection model
yolo_model = YOLO('detector/model_weights/yolo_text_detector.pt')

# --- FIX for model loading ---
import detector.crnn_model
detector.crnn_model.CRNN = CRNNModel
sys.modules['__main__'] = detector.crnn_model
# --- End of model loading fix ---

# Load the model object
crnn_model = torch.load(
    'detector/model_weights/best_crnn_mozhi_model.pt',
    map_location=device,
    weights_only=False
)

# --- FINAL FIX for AttributeError STARTS HERE ---
# Patch the loaded model to align its structure with the new code.
# The loaded object has an 'rnn' attribute which IS the LSTM layer.
# Our new code expects this layer to be named 'lstm'. We just rename it.
if hasattr(crnn_model, 'rnn') and not hasattr(crnn_model, 'lstm'):
    print("Patching loaded model: Renaming 'rnn' attribute to 'lstm'.")
    crnn_model.lstm = crnn_model.rnn
    del crnn_model.rnn
# --- FINAL FIX for AttributeError ENDS HERE ---


crnn_model.to(device)
crnn_model.eval()

# Define preprocessing pipeline for CRNN input images (expects 3 channels)
crnn_transform = transforms.Compose([
    transforms.Resize((32, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess_crnn_image(img):
    """Preprocess cropped PIL image for CRNN input tensor."""
    return crnn_transform(img).unsqueeze(0).to(device)

def decode_crnn_output(output):
    """Decode the CRNN output tensor to string."""
    char_list = "0123456789abcdefghijklmnopqrstuvwxyz"
    
    _, preds = output.max(2)
    preds = preds.squeeze(0).cpu().numpy()

    blank_idx = len(char_list)
    decoded = []
    prev = blank_idx
    for p in preds:
        if p != prev and p != blank_idx:
            if p < len(char_list):
                 decoded.append(char_list[p])
        prev = p
    return ''.join(decoded)

def detect_and_recognize_text(image_path):
    """Detect text boxes from image using YOLO and recognize text using CRNN."""
    img = cv2.imread(image_path)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    results = yolo_model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    recognized_texts = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = pil_img.crop((x1, y1, x2, y2))
        input_tensor = preprocess_crnn_image(crop)
        with torch.no_grad():
            output = crnn_model(input_tensor)

        text = decode_crnn_output(output)
        recognized_texts.append({'box': (x1, y1, x2, y2), 'text': text})

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    filename_base = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{filename_base}_detected.jpg"
    output_path = os.path.join(settings.MEDIA_ROOT, output_filename)
    cv2.imwrite(output_path, img)

    return {
        'output_image': output_filename,
        'recognized_texts': recognized_texts
    }