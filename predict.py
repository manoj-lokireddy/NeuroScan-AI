import sys
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/brain_tumor_best.keras"
IMG_SIZE = (224, 224)

# Check image path
if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load and preprocess image
image = tf.keras.utils.load_img(
    image_path,
    target_size=IMG_SIZE
)

image_array = tf.keras.utils.img_to_array(image)
image_array = image_array / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Prediction
prediction = model.predict(image_array, verbose=0)[0][0]

if prediction >= 0.5:
    result = "TUMOR"
    confidence = prediction * 100
else:
    result = "NO TUMOR"
    confidence = (1 - prediction) * 100

print("\n================================")
print("      BRAIN TUMOR PREDICTION")
print("================================")
print(f"Image      : {image_path}")
print(f"Prediction : {result}")
print(f"Confidence : {confidence:.2f}%")
print("================================")