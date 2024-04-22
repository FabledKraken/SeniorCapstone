import os
import cv2
import numpy as np

from utils import get_face_landmarks

data_dir = './data'

output = []

# Loop through each emotion category and its index
for emotion_index, emotion in enumerate(sorted(os.listdir(data_dir))):
    # Get the list of images in the current emotion category
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        # Construct the full path to the image
        image_path = os.path.join(data_dir, emotion, image_path_)

        # Read the image from the path using OpenCV
        image = cv2.imread(image_path)

        # Extract facial landmarks from the image using the written function from utils.py
        face_landmarks = get_face_landmarks(image)

        # Ensure that a face is detected by checking the length of landmarks
        # Expected number of landmarks is 1404; if not, it means no face was detected
        # Uncomment for debugging: print(len(face_landmarks))

        # If the expected number of landmarks is found, assume a face was detected
        if len(face_landmarks) == 1404:
            # Append the emotion index to the list of landmarks (as the label)
            face_landmarks.append(int(emotion_index))
            # Add the landmarks with the appended emotion label to the output list
            output.append(face_landmarks)

# Save data file with readable format
np.savetxt('data.txt', np.asarray(output))
