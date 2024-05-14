"""
DeepFace - Example
"""

import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from platform import system

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QInputDialog, QLabel, QMainWindow

from deepface import DeepFace
from deepface.modules import streaming
from ui.main_window import Ui_MainWindow

FPS: int = 20
INTERVAL: int = int(1000 / FPS)
DATABASE_PATH: str = "database"

MODEL_NAME: str = "Facenet512"

REGISTER_FPS: int = 30
REGISTER_INTERVAL: int = int(1000 / FPS)


class ButtonState(Enum):
    """
    An enumeration representing the state of a button.

    Attributes:
        INACTIVE (int): The state representing a button that is not currently being interacted with.
        ACTIVE (int): The state representing a button that is currently being interacted with.
    """

    INACTIVE = 0
    ACTIVE = 1


@dataclass
class IdentityResult:
    """
    A dataclass representing the result of an identity search.

    Attributes:
        identified_image_path (str): The path to the identified image.
        identified_image (np.ndarray | None): The identified image as a numpy array, or None if no image was found.
        distance (float): The distance between the identified face and the face in the database.
        threshold (float): The threshold for considering a match.
    """

    identified_image_path: str
    identified_image: np.ndarray | None
    distance: float
    threshold: float


class DeepFaceHelper:
    """
    A helper class for handling facial recognition tasks using the DeepFace library.

    Args:
        db_path (str): The path to the database of faces to be recognized.
        model_name (str): The name of the model to be used for face recognition.
        detector_backend (str): The backend to be used for face detection.
        distance_metric (str): The metric to be used for measuring similarity.
        source (int): The source of the video capture.
        frame_threshold (int): The threshold for the number of frames with faces.
    """

    def __init__(
        self,
        db_path,
        model_name="VGG-Face",
        detector_backend="opencv",
        distance_metric="cosine",
        source=0,
        frame_threshold=30,
    ):
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.frame_threshold = frame_threshold
        if system() == "Windows":
            self.source = source + cv2.CAP_DSHOW  # default 0 + Windows DirectShow
        else:
            self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.num_frames_with_faces = 0
        self.register_img = None

    def face_analysis(self) -> None:
        """
        This method is responsible for building the facial recognition model and initializing the embeddings.

        Note: The detected face is a 224x224x3 numpy array of zeros, which serves as a placeholder for the actual face data.
        """
        streaming.build_facial_recognition_model(model_name=self.model_name)

        # call a dummy find function for db_path once to create embeddings before start capturing
        _ = DeepFaceHelper.search_identity_details(
            detected_face=np.zeros([224, 224, 3]),  # random numbers
            db_path=self.db_path,
            detector_backend=self.detector_backend,
            distance_metric=self.distance_metric,
            model_name=self.model_name,
        )

    def capture_camera_and_analyze_face(self, face_found: bool) -> None:
        """
        Capture camera and analyze face.

        This method captures the frame from the camera and performs face analysis on it.
        If a face is found, it performs facial recognition and updates the status bar with the identity result.

        Args:
            face_found (bool): A flag indicating whether a face has been found in the previous frame.

        Returns:
            None
        """
        has_frame, img = self.cap.read()
        if not has_frame:
            logging.error("Could not read frame from camera")
            return

        raw_img = img.copy()

        if not face_found:
            faces_coordinates = streaming.grab_facial_areas(
                img, detector_backend=self.detector_backend
            )

            detected_faces = streaming.extract_facial_areas(
                img=img, faces_coordinates=faces_coordinates
            )
            img = streaming.highlight_facial_areas(
                img=img, faces_coordinates=faces_coordinates
            )

            self.num_frames_with_faces = (
                self.num_frames_with_faces + 1 if len(faces_coordinates) else 0
            )

            face_found = (
                self.num_frames_with_faces > 0
                and self.num_frames_with_faces % self.frame_threshold == 0
            )
            if face_found:
                # add analyze results into img - derive from raw_img
                img = streaming.highlight_facial_areas(
                    img=raw_img, faces_coordinates=faces_coordinates
                )

                identity = self.perform_facial_recognition_details(
                    img=img,
                    faces_coordinates=faces_coordinates,
                    detected_faces=detected_faces,
                    db_path=self.db_path,
                    detector_backend=self.detector_backend,
                    distance_metric=self.distance_metric,
                    model_name=self.model_name,
                )

                if not identity:
                    logging.info("No identity found.")
                    window.statusbar_identity_result.setText("No identity found.")
                    window.statusbar_identity_result.setStyleSheet("color: red;")
                    face_found = False
                    self.num_frames_with_faces = 0
                else:
                    img = identity.identified_image
                    identity_path_parts = identity.identified_image_path.split(os.sep)
                    short_identity_path = os.path.join(
                        identity_path_parts[1], identity_path_parts[2][:5]
                    )
                    window.statusbar_identity_result.setText(
                        f"Identity: {short_identity_path} | Distance: {identity.distance:.2f} | Threshold: {identity.threshold:.2f}"
                    )
                    window.statusbar_identity_result.setStyleSheet("color: green;")
                    window.stop_verify_identified()
                    QTimer.singleShot(
                        200, lambda: window.show_opencv_img(img)
                    )  # Ensure that right image is shown (QTimer act asynchronously)

        window.show_opencv_img(img)

    @staticmethod
    def search_identity_details(
        detected_face: np.ndarray,
        db_path: str,
        model_name: str,
        detector_backend: str,
        distance_metric: str,
    ) -> IdentityResult | None:
        """
        Search an identity in facial database.

        Args:
            detected_face (np.ndarray): extracted individual facial image
            db_path (string): Path to the folder containing image files. All detected faces
                in the database will be considered in the decision-making process.
            model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
                OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
            detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
                'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
                (default is opencv).
            distance_metric (string): Metric for measuring similarity. Options: 'cosine',
                'euclidean', 'euclidean_l2' (default is cosine).

        Returns:
            identity_result (IdentityResult): identified image path, identified image, distance and threshold
        """
        try:
            dfs = DeepFace.find(
                img_path=detected_face,
                db_path=db_path,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                enforce_detection=False,
                silent=True,
            )
        except ValueError as err:
            if f"No item found in {db_path}" in str(err):
                logging.warning(
                    f"No item is found in {db_path}."
                    "So, no facial recognition analysis will be performed."
                )
                dfs = []
            else:
                raise err
        if len(dfs) == 0:
            return None

        # detected face is coming from parent, safe to access 1st index
        df = dfs[0]
        if df.shape[0] == 0:
            return None

        candidate = df.iloc[0]
        identity_result = IdentityResult(
            identified_image_path=candidate["identity"],
            identified_image=None,
            distance=candidate["distance"],
            threshold=candidate["threshold"],
        )

        # load found identity image - extracted if possible
        target_objs = DeepFace.extract_faces(
            img_path=identity_result.identified_image_path,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True,
        )

        # extract facial area of the identified image if and only if it has one face
        # otherwise, show image as is
        if len(target_objs) == 1:
            # extract 1st item directly
            target_obj = target_objs[0]
            target_img = target_obj["face"]
            target_img = cv2.resize(
                target_img,
                (streaming.IDENTIFIED_IMG_SIZE, streaming.IDENTIFIED_IMG_SIZE),
            )
            target_img *= 255
            target_img = target_img[:, :, ::-1]
        else:
            target_img = cv2.imread(identity_result.identified_image_path)

        identity_result.identified_image = target_img

        return identity_result

    def perform_facial_recognition_details(
        self,
        img,
        faces_coordinates,
        detected_faces,
        db_path=None,
        detector_backend=None,
        distance_metric=None,
        model_name=None,
    ) -> IdentityResult | None:
        """
        Perform facial recognition on the detected faces.

        This method takes an image, the coordinates of the faces in the image, and the detected faces themselves,
        and performs facial recognition on each face. It uses the specified database path, detector backend,
        distance metric, and model name, or defaults to the ones specified in the class instance if not provided.

        Args:
            img (np.ndarray): The image to perform facial recognition on.
            faces_coordinates (list): A list of tuples containing the coordinates of each face in the image.
            detected_faces (list): A list of the detected faces as numpy arrays.
            db_path (str, optional): The path to the database of faces to be recognized. Defaults to the one specified in the class instance.
            detector_backend (str, optional): The backend to be used for face detection. Defaults to the one specified in the class instance.
            distance_metric (str, optional): The metric to be used for measuring similarity. Defaults to the one specified in the class instance.
            model_name (str, optional): The name of the model to be used for face recognition. Defaults to the one specified in the class instance.

        Returns:
            IdentityResult | None: An instance of the IdentityResult dataclass containing the identified image path, the identified image, the distance, and the threshold, or None if no identity was found.
        """
        db_path = db_path or self.db_path
        detector_backend = detector_backend or self.detector_backend
        distance_metric = distance_metric or self.distance_metric
        model_name = model_name or self.model_name

        for idx, (x, y, w, h) in enumerate(faces_coordinates):
            detected_face = detected_faces[idx]
            identity_result = DeepFaceHelper.search_identity_details(
                detected_face=detected_face,
                db_path=db_path,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                model_name=model_name,
            )
            if not identity_result:
                continue

            identity_result.identified_image = streaming.overlay_identified_face(
                img=img,
                target_img=identity_result.identified_image,
                label=identity_result.identified_image_path.split(os.sep)[1],
                x=x,
                y=y,
                w=w,
                h=h,
            )

        return identity_result

    def capture_camera(self):
        """
        Capture a frame from the camera and display it.

        Returns:
            None
        """
        has_frame, self.register_img = self.cap.read()
        if not has_frame:
            logging.error("Could not read frame from camera")

        window.show_opencv_img(self.register_img)

    def load_registered(self):
        """
        Load the names of the registered faces from the database.

        This method lists the names of the directories in the database path, each of which represents a registered face.

        Returns:
            list: A list of the names of the registered faces.
        """
        return [
            name
            for name in os.listdir(DATABASE_PATH)
            if os.path.isdir(os.path.join(DATABASE_PATH, name))
        ]

    def register_face(self, folder_name: str):
        """
        Register a face in the database.

        This method takes a folder name as input and checks if a directory with that name exists in the database.
        If it doesn't, it creates one. Then, it encodes the currently captured image (stored in `self.register_img`)
        into PNG format and calculates its SHA-256 hash. Finally, it saves the image in the newly created directory,
        using the hash as the file name.

        Args:
            folder_name (str): The name of the folder to create in the database.

        Returns:
            None
        """
        if not os.path.exists(os.path.join(DATABASE_PATH, folder_name)):
            os.makedirs(os.path.join(DATABASE_PATH, folder_name))

        _, binary_data = cv2.imencode(".png", self.register_img)

        sha256_hash = hashlib.sha256(binary_data).hexdigest()

        cv2.imwrite(
            os.path.join(DATABASE_PATH, folder_name, f"{sha256_hash}.png"),
            self.register_img,
        )


class MainWindow(QMainWindow):
    """
    The MainWindow class inherits from QMainWindow and represents the main window of the application.

    This class is responsible for setting up the user interface, handling user interactions, and managing the application's state.
    It contains methods for handling button click events, displaying images, and managing the state of the application.

    """

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.verify_pushbutton.clicked.connect(self.verify_pushbutton_click)
        self.ui.register_pushbutton.clicked.connect(self.register_pushbutton_click)
        self.ui.registered_pushbutton.clicked.connect(self.registered_pushbutton_click)
        self.ui.take_picture_pushbutton.clicked.connect(
            self.take_picture_pushbutton_click
        )
        self.ui.register_picture_pushbutton.clicked.connect(
            self.register_picture_pushbutton_click
        )
        self.reset_camera_label()
        self.verify_state = ButtonState.INACTIVE
        self.register_state = ButtonState.INACTIVE
        self.take_picture_state = ButtonState.INACTIVE
        self.ui.register_frame.hide()

        self.deep_face_helper = DeepFaceHelper(DATABASE_PATH)
        self.face_analysis_timer = QTimer(self)
        self.face_analysis_timer.timeout.connect(
            lambda: self.deep_face_helper.capture_camera_and_analyze_face(False)
        )
        self.register_timer = QTimer(self)
        self.register_timer.timeout.connect(self.deep_face_helper.capture_camera)

        self.statusbar_model = QLabel(f"Model: {self.deep_face_helper.model_name}")
        self.statusbar_detector_backend = QLabel(
            f"Detector: {self.deep_face_helper.detector_backend}"
        )
        self.statusbar_model.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.statusbar_detector_backend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.statusbar.addWidget(self.statusbar_model, stretch=1)
        self.ui.statusbar.addWidget(self.statusbar_detector_backend, stretch=1)

        self.statusbar_identity_result = QLabel("")
        self.statusbar_identity_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.statusbar.addPermanentWidget(self.statusbar_identity_result, stretch=6)
        self.ui.statusbar.setStyleSheet(
            "margin-right: 0px; padding: 3px; font-size: 10pt;"
        )

    def show_opencv_img(self, img):
        """
        Display an OpenCV image in the GUI.

        Args:
            img (np.ndarray): The OpenCV image to be displayed.

        Returns:
            None
        """
        convert = QImage(img, img.shape[1], img.shape[0], QImage.Format.Format_BGR888)
        self.ui.camera_label.setPixmap(QPixmap.fromImage(convert))

    def reset_verify_button(self):
        """
        Reset the verify button to its default state.

        Returns:
            None
        """
        self.ui.verify_pushbutton.setText("Verify")
        self.ui.verify_pushbutton.setStyleSheet("QPushButton {color: white;}")

    def reset_register_button(self):
        """
        Reset the register button to its default state.

        Returns:
            None
        """
        self.ui.register_pushbutton.setText("Register")
        self.ui.register_pushbutton.setStyleSheet("QPushButton {color: white;}")

    def reset_register_frame(self):
        """
        Hide the register frame in the GUI.

        Returns:
            None
        """
        self.ui.register_frame.hide()

    def reset_camera_label(self):
        """
        Reset the camera label to its default state.

        Returns:
            None
        """
        self.ui.camera_label.setPixmap(
            QPixmap("./ui/resources/deepface-icon-labeled.png")
        )

    def start_verify_manually(self):
        """
        Start the manual verification process.

        Returns:
            None
        """
        self.deep_face_helper.face_analysis()

        self.face_analysis_timer.start(INTERVAL)

        self.verify_state = ButtonState.ACTIVE
        self.ui.verify_pushbutton.setText("Cancel Verification")
        self.ui.verify_pushbutton.setStyleSheet("QPushButton {color: yellow;}")
        self.statusbar_identity_result.setText("")

    def stop_verify_identified(self):
        """
        Stop the verification process after an identity has been found.

        Returns:
            None
        """
        self.face_analysis_timer.stop()
        self.verify_state = ButtonState.INACTIVE
        self.reset_verify_button()

    def stop_verify_manually(self):
        """
        Stop the manual verification process.

        Returns:
            None
        """
        self.face_analysis_timer.stop()
        self.verify_state = ButtonState.INACTIVE
        self.reset_verify_button()
        self.reset_camera_label()
        self.statusbar_identity_result.setText("")

    def verify_pushbutton_click(self):
        """
        Handle the click event of the verify button.

        Returns:
            None
        """
        if self.verify_state == ButtonState.INACTIVE:
            self.start_verify_manually()
        elif self.verify_state == ButtonState.ACTIVE:
            self.stop_verify_manually()

    def register_pushbutton_click(self):
        """
        Handle the click event of the register button.

        Returns:
            None
        """
        if self.register_state == ButtonState.INACTIVE:
            self.register_timer.start(REGISTER_INTERVAL)
            self.register_state = ButtonState.ACTIVE
            self.ui.register_pushbutton.setText("Stop registering")
            self.ui.register_pushbutton.setStyleSheet("QPushButton {color: red;}")
            self.ui.register_frame.show()
        elif self.register_state == ButtonState.ACTIVE:
            self.register_timer.stop()
            self.reset_camera_label()
            self.register_state = ButtonState.INACTIVE
            self.reset_register_button()
            self.ui.register_frame.hide()
        self.statusbar_identity_result.setText("")

    def take_picture_pushbutton_click(self):
        """
        Handle the click event of the take picture button.

        Returns:
            None
        """
        if self.take_picture_state == ButtonState.INACTIVE:
            self.register_timer.stop()
            self.take_picture_state = ButtonState.ACTIVE
            self.ui.take_picture_pushbutton.setText("Retake picture")
            self.ui.register_picture_pushbutton.setEnabled(True)
            self.ui.take_picture_pushbutton.setStyleSheet(
                "QPushButton {color: yellow;}"
            )
        elif self.take_picture_state == ButtonState.ACTIVE:
            self.register_timer.start(REGISTER_INTERVAL)
            self.take_picture_state = ButtonState.INACTIVE
            self.ui.take_picture_pushbutton.setText("Take picture")
            self.ui.register_picture_pushbutton.setEnabled(False)

    def register_picture_pushbutton_click(self):
        """
        Handle the click event of the register picture button.

        This method loads the names of the registered faces from the database and displays a dialog for the user to choose or input an option.
        If the user confirms their choice or input, the method registers the face under the chosen or inputted name.

        Returns:
            None
        """
        options = self.deep_face_helper.load_registered()

        option, ok = QInputDialog.getItem(
            self, "Register Dialog", "Choose or input an option:", options, 0, True
        )
        if ok and option:
            self.deep_face_helper.register_face(option)

    def registered_pushbutton_click(self):
        """
        Handle the click event of the registered button.

        This method opens the directory containing the database of registered faces using the default file explorer of the operating system.

        Returns:
            None
        """
        os.startfile(DATABASE_PATH)


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
