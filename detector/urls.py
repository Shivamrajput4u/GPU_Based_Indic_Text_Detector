from django.urls import path
from . import views

urlpatterns = [
    # Home page route — displays upload form or index page
    path('', views.index, name='index'),

    # Text detection route — handles image upload and inference
    path('detect/', views.detect_text, name='detect_text'),
]
