"""
Car model classifier trained with Google Teachable Machine.

Export from Teachable Machine: Export Model > Tensorflow > Keras > Download my model.
Unzip and put both files in the model/ folder:
    model/keras_model.h5
    model/labels.txt

The classifier is optional: if the files are missing, the app still works and the user
simply picks their car model from the dropdown.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "model"))
MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"
IMAGE_SIZE = (224, 224)  # Teachable Machine image models use 224 x 224 input


def parse_labels(lines):
    """Teachable Machine writes labels as '0 Myvi', '1 Axia' ... -> ['Myvi', 'Axia', ...]."""
    labels = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        first, _, rest = line.partition(" ")
        labels.append(rest.strip() if first.isdigit() and rest.strip() else line)
    return labels


def preprocess(photo: Image.Image) -> np.ndarray:
    """Same preprocessing as Teachable Machine's own sample code."""
    image = ImageOps.fit(photo.convert("RGB"), IMAGE_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(image).astype(np.float32)
    normalized = (array / 127.5) - 1.0
    return normalized[np.newaxis, ...]  # shape (1, 224, 224, 3)


class CarClassifier:
    def __init__(self, model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.model = None
        self.labels = []
        self.error = None
        self._tried = False

    def _load(self):
        if self._tried:
            return
        self._tried = True

        if not self.model_path.exists() or not self.labels_path.exists():
            self.error = "No Teachable Machine model found in the model/ folder yet."
            return
        try:
            # Teachable Machine saves Keras 2 (.h5) models. TensorFlow 2.16+ ships Keras 3 by default,
            # so we use the tf-keras package (Keras 2) when it is installed.
            try:
                import tf_keras as keras
            except ImportError:
                import tensorflow as tf

                keras = tf.keras

            # Teachable Machine exports include a 'groups' setting that some Keras versions reject.
            # This small wrapper drops it so the model loads.
            class FixedDepthwiseConv2D(keras.layers.DepthwiseConv2D):
                def __init__(self, *args, groups=None, **kwargs):
                    super().__init__(*args, **kwargs)

            self.model = keras.models.load_model(
                str(self.model_path),
                compile=False,
                custom_objects={"DepthwiseConv2D": FixedDepthwiseConv2D},
            )
            self.labels = parse_labels(self.labels_path.read_text(encoding="utf-8").splitlines())
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.error = f"Could not load the car classifier: {exc}"
            print(f"[classifier] {self.error}")

    @property
    def available(self) -> bool:
        self._load()
        return self.model is not None

    def predict(self, photo: Image.Image, top_k: int = 3) -> dict:
        """Return {label: confidence} for the top_k classes, highest first."""
        if not self.available:
            return {}
        scores = self.model.predict(preprocess(photo), verbose=0)[0]
        ranked = sorted(zip(self.labels, scores), key=lambda pair: float(pair[1]), reverse=True)
        return {label: float(score) for label, score in ranked[:top_k]}
