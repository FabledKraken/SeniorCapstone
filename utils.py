import cv2
import mediapipe as mp


# Function with parameters image, draw (whether to draw detected landmarks on the image),
#   and static_image_mode (True=image, False=video)
def get_face_landmarks(image, draw=False, static_image_mode=True):
    # Read the input image and converts color from BGR (used by OpenCV) to RGB for Mediapipe
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initializes teh MediaPipe FaceMesh model with specified parameters
    #   static_image_mode: True if using images and False if using video
    #   max_num_faces: maximum number of detected faces detected at a time
    #   min_detection_confidence: confidence threshold for detecting faces
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                                max_num_faces=1,
                                                min_detection_confidence=0.5)
    # image.shape returns a tuple (height, width, channels) which is rows, columns, and depth respectively. The depth
    #   is usually 3 for RGB. image_rows = height of image, image_col = width of image, and _ is used as a placeholder
    #   in Python for unwanted values
    image_rows, image_cols, _ = image.shape

    # Processes the RGB image to detect facial landmarks
    results = face_mesh.process(image_input_rgb)

    # Initializes a list to store normalized landmark coordinates
    normalized_landmarks = []

    # Checks if any faces are detected
    if results.multi_face_landmarks:

        # if=True, use MediaPipes drawing utilities to visualize the landmarks and their connections on the image
        if draw:
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=0)

            ### USED FOR CUSTOMIZING THE MESH DRAWING (WORKING CODE)
            # # Customize drawing specification for landmarks and connections
            # landmark_drawing_spec = mp_drawing_styles.get_default_face_mesh_landmarks_style()
            # connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()

            # Draws the detected facial landmarks on the first face detected [0]
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        # Extracts the landmarks of the face detected (nose tip, corner of eye, etc.)
        ls_single_face = results.multi_face_landmarks[0].landmark

        # Initialize lists to store landmark coordinates
        x_coordinates, y_coordinates, z_coordinates = [], [], []

        # Collect coordinates from each landmark on the face
        for landmark in ls_single_face:
            x_coordinates.append(landmark.x)
            y_coordinates.append(landmark.y)
            z_coordinates.append(landmark.z)

        # Calculate minimum values
        min_x = min(x_coordinates)
        min_y = min(y_coordinates)
        min_z = min(z_coordinates)

        # Normalizes the coordinates by subtracting min from each dimensional value
        #   very important to normalize (smallest value = 0, making subsequent processing steps more consistent
        #   and more accurate. Important to have data start at a common starting point
        for i in range(len(x_coordinates)):
            normalized_landmarks.extend([
                x_coordinates[i] - min_x,
                y_coordinates[i] - min_y,
                z_coordinates[i] - min_z
            ])

    return normalized_landmarks
