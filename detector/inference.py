import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import easyocr
from django.conf import settings

# Set the device to 'cpu'. Note: A GPU would be much faster.
device = 'cpu'

# Define the list of compatible languages for EasyOCR.
# 'ur' (Urdu) was removed as it uses an Arabic script, which is not
# compatible with Indic scripts like 'hi' and 'mr' in the same instance.
EASYOCR_SAFE_LANGS = [
    'en',  # English
    'hi',  # Hindi
    'mr',  # Marathi
]

# Initialize the models
# The YOLO model detects the location of text boxes.
yolo_model = YOLO('detector/model_weights/yolo_text_detector.pt')
# The EasyOCR reader recognizes the characters within the detected boxes.
ocr_reader = easyocr.Reader(EASYOCR_SAFE_LANGS, gpu=False)

def detect_and_recognize_text(image_path):
    """
    Detects text boxes in an image using YOLO and recognizes the text
    within those boxes using EasyOCR.

    Args:
        image_path (str): The full path to the input image file.

    Returns:
        dict: A dictionary containing the path to the output image with
              bounding boxes and the recognized text data.
    """
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    # Convert the image from BGR (OpenCV format) to RGB for PIL
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Perform text detection using the YOLO model
    results = yolo_model(img)
    # Get the bounding box coordinates [x1, y1, x2, y2]
    boxes = results[0].boxes.xyxy.cpu().numpy()

    recognized_texts = []

    # Loop through each detected bounding box
    for box in boxes:
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, box)

        # Crop the detected region from the PIL image
        crop = pil_img.crop((x1, y1, x2, y2))
        # Convert the cropped PIL image back to an OpenCV image (numpy array)
        crop_np = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)

        # Perform OCR on the cropped image
        ocr_results = ocr_reader.readtext(crop_np)

        # Join the recognized text pieces into a single string
        text = " ".join([res[1] for res in ocr_results]) if ocr_results else ""
        print(f"Box: {(x1, y1, x2, y2)} => Text: {text}")

        # Store the results
        recognized_texts.append({'box': (x1, y1, x2, y2), 'text': text})

        # Draw the bounding box on the original image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put the recognized text above the bounding box
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    # Create a new filename for the output image
    filename_base = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{filename_base}_detected.jpg"
    output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

    # Save the image with the drawn boxes and text
    cv2.imwrite(output_path, img)

    # Return the path to the new image and the extracted text data
    return {
        'output_image': output_filename,
        'recognized_texts': recognized_texts
    }