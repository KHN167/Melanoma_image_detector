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
        img = Image.open(uploaded_image)
        img.verify()  # Verifies the image data
        return True
    except (IOError, SyntaxError) as e:
        return False

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        # Process the uploaded image here

        model_path = os.path.join(settings.BASE_DIR, 'benignmalignant.h5')
        model = load_model(model_path)

        img = Image.open(uploaded_image)
        


        if is_valid_image(img):
            with Image.open(uploaded_image) as img:
                img1 = np.array(img)

                resize = cv2.resize(img1, (256, 256))
                resize = tf.image.resize(resize, (256, 256))

                yhat = model.predict(np.expand_dims(resize/255, 0))

                if yhat > 0.5:
                    predicted_class = 'malignant'
                else:
                    predicted_class = 'benign'

        else:
            return HttpResponse('invalid image file. please upload a valid image.')






        # Save the image to a directory
        image_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_images', uploaded_image.name)
        with open(image_path, 'wb') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        return render(request, 'results.html', {'predicted_class': predicted_class, 'score': yhat[0][0]})

    return render(request, 'index.html')