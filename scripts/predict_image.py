import numpy as np
import requests
from PIL import Image
import base64
from io import BytesIO
import sys

# Load and preprocess the image
img_path = sys.argv[1]  
img = Image.open(img_path).convert('L')  # Convert to grayscale
img = img.resize((28, 28))  # Resize to 28x28 pixels
image_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
image_array = image_array.reshape(28, 28, 1)  # Add batch dimension

# Prepare the JSON payload with normalized image data
payload = {
    "instances": [
        {"conv2d_input": image_array.tolist()}
    ]
}

url = "http://localhost:8000/v1/models/digits-recognizer:predict"

# Send the request
response = requests.post(url, json=payload)

# Check the response
print(response.json())