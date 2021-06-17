import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image


def response_from_server(url, image_file):
    """Makes a POST request to the server and returns the response.

    Args:
        url (str): URL that the request is sent to.
        image_file (_io.BufferedReader): File to upload, should be an image.

    Returns:
        requests.models.Response: Response from the server.
    """

    files = {"file": image_file}
    response = requests.post(url, files=files)
    status_code = response.status_code
    if status_code != 200:
        print("SOMETHING WENT WRONG!!!")
    return response


def display_image_from_response(response):
    """Display image within server's response.

    Args:
        response (requests.models.Response): The response from the server after object detection.
    """

    image_stream = io.BytesIO(response.content)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    filename = "predictions.jpg"
    cv2.imwrite(f"server_response/{filename}", image)
    image = cv2.imread(f"server_response/{filename}")
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    plt.show()
