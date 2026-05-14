# 🌿 Plant Disease Prediction Script

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# -----------------------------
# 1. Load trained model
# -----------------------------
model_path = "plant_disease_model.keras"

model = load_model(model_path, compile=False)

print("Model loaded successfully ✔️")

# -----------------------------
# 2. Class names (IMPORTANT)
# -----------------------------
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# -----------------------------
# 3. Image path (CHANGE THIS)
# -----------------------------
img_path = "test_images/tomatoleaf.jpg"

# Check if file exists
if not os.path.exists(img_path):
    print("Image not found ❌ Please check path")
    exit()

# -----------------------------
# 4. Load and preprocess image
# -----------------------------
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# -----------------------------
# 5. Prediction
# -----------------------------
prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]

confidence = np.max(prediction) * 100

# -----------------------------
# 6. Output result
# -----------------------------
print("\n🌿 Prediction Result")
print("----------------------")
print("Disease:", predicted_class)
print("Confidence:", round(confidence, 2), "%")