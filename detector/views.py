from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings  # <-- 1. IMPORT SETTINGS
from django.http import JsonResponse
from .inference import detect_and_recognize_text
import os

def index(request):
    """Handles the initial page load and displays the upload form."""
    return render(request, 'detector/index.html')

def detect_text(request):
    """Handles the image upload, processing, and displays the result."""
    context = {}
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()

        # Save the uploaded image to the media directory
        filename = fs.save(image.name, image)
        abs_path = fs.path(filename)

        # Run your YOLO + EasyOCR inference
        result = detect_and_recognize_text(abs_path)

        # --- 2. THIS IS THE CRITICAL FIX ---
        # Create the full, correct URL for the browser to use.
        # For example: '/media/' + 'my_image_detected.jpg'
        output_image_url = os.path.join(settings.MEDIA_URL, result['output_image'])
        
        # Prepare context for rendering the result
        context['output_image_url'] = output_image_url
        context['recognized_texts'] = result.get('recognized_texts')

        # This part seems fine, keeping your language detection logic
        try:
            from langdetect import detect
            combined_text = ' '.join([item['text'] for item in context['recognized_texts'] if item['text'].strip()])
            if combined_text:
                context['detected_language'] = detect(combined_text)
            else:
                context['detected_language'] = 'unknown'
        except (ImportError, Exception):
            context['detected_language'] = 'unknown'

        # Render the same page, but now with the context data for the results
        return render(request, 'detector/index.html', context)

    # If not a POST request, just show the upload page
    return render(request, 'detector/index.html')