# Indic Text Detector 🖼🔍

An end-to-end web application built with Django and PyTorch that detects and recognizes text from images, with a focus on English and Indic scripts like Hindi and Marathi. The application also provides a prediction for the primary language of the extracted text.



## ✨ Features

- *Simple Web Interface:* Easy-to-use interface for uploading images.
- *Text Detection:* Utilizes a *YOLO (You Only Look Once)* model to accurately locate text boxes within an image.
- *Text Recognition (OCR):* Employs *EasyOCR* to read and transcribe the text from the detected boxes.
- *Multi-Language Support:* Configured to recognize English, Hindi, and Marathi text.
- *Language Prediction:* Uses the langdetect library to predict the language of the combined recognized text.
- *Dynamic Results:* Displays the processed image with bounding boxes drawn around the text, alongside a list of extracted text.

## 🛠 Tech Stack

- *Backend:* Python, Django
- *Machine Learning:* PyTorch, Ultralytics YOLO, EasyOCR
- *Frontend:* HTML, Tailwind CSS
- *Core Libraries:* OpenCV, Pillow, langdetect

## 🚀 Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

- Python 3.10+
- Git

### Installation Steps

1.  *Clone the repository:*
    bash
    git clone <your-repository-url>
    cd indic-text-detector
    

2.  *Create and activate a virtual environment:*
    bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    

3.  *Install the required dependencies:*
    bash
    pip install -r requirements.txt
    

4.  *Download Model Weights:*
    This project requires pre-trained model weights for text detection.
    - Create a directory: mkdir -p detector/model_weights/
    - Download the yolo_text_detector.pt file and place it inside the detector/model_weights/ directory.

5.  *Run Django Migrations:*
    bash
    python manage.py migrate
    

6.  *Start the development server:*
    bash
    python manage.py runserver
    
    The application will be available at http://127.0.0.1:8000/.

##  usage

1.  Navigate to the homepage.
2.  Click "Choose file" to select an image containing text.
3.  Click the "Analyze →" button to submit.
4.  The page will reload, displaying the result with the processed image, a list of extracted text, and the predicted language.

## 📁 Project Structure

indic-text-detector/ ├── detector/ # Main Django App │ ├── migrations/ │ ├── model_weights/ │ │ └── yolo_text_detector.pt (You must add this) │ ├── templates/ │ │ └── detector/ │ │ └── index.html │ ├── init.py │ ├── admin.py │ ├── apps.py │ ├── inference.py # Core ML inference logic │ ├── models.py │ ├── tests.py │ ├── urls.py │ └── views.py # Handles web requests ├── indic_text_detector/ # Django Project Configuration │ ├── init.py │ ├── asgi.py │ ├── settings.py │ ├── urls.py │ └── wsgi.py ├── media/ # Stores user-uploaded images ├── manage.py ├── requirements.txt └── README.md

