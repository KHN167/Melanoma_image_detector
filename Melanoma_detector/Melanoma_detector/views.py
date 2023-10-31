from django.shortcuts import render
import os
from django.conf import settings
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
from django.http import HttpResponse



def index(request):
    return render(request, 'index.html')


def is_valid_image(uploaded_image):
    try:
        # Attempt to open the image
        with Image.open(uploaded_image) as img:
            return True
    except:
        return False

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        # Process the uploaded image here


        model_path = os.path.join(settings.BASE_DIR, 'benignmalignant.h5')
        model = load_model(model_path)

        img = Image.open(uploaded_image)
        


        with Image.open(uploaded_image) as img:
            img = img.convert('RGB')
            img_array = np.array(img)

            # Resize the image
            img_array = cv2.resize(img_array, (256, 256))
            img_array = img_array / 255.0  # Normalize the image

            # Make predictions
            predictions = model.predict(np.expand_dims(img_array, axis=0))

            # Process the predictions
            if predictions > 0.5:
                predicted_class = 'malignant'
                score = predictions[0][0] * 100
            else:
                predicted_class = 'benign'
                score = 100 - predictions[0][0] 

        return render(request, 'results.html', {'predicted_class': predicted_class, 'score': score})

    return render(request, 'index.html')