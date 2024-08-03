import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained model
model_path = 'mymodel.h5'
model = load_model(model_path)

# Function to preprocess and classify an image
def classify_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((150, 150))  # Resize image to match model's expected sizing
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to fit the expected model input
        img_array = preprocess_input(img_array)  # Preprocess the image

        # Predict the class
        predictions = model.predict(img_array)
        if predictions[0][0] > 0.5:
            return "cancerous"
        else:
            return "non-cancerous"
    except Exception as e:
        print(f"Error classifying image: {e}")
        return "error"

# Function to handle file upload and classification
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = classify_image(file_path)
        result_label.config(text=f"Prediction: {result}")
        # Display the uploaded image
        img = Image.open(file_path)
        img = img.resize((250, 250))  # Resize image for display
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img
    else:
        messagebox.showerror("Error", "No file selected.")

# Create the main window
window = tk.Tk()
window.title("Leukemia Detection")
window.geometry("400x400")

# Create UI components
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

result_label = tk.Label(window, text="Prediction: ")
result_label.pack()

img_label = tk.Label(window)
img_label.pack(pady=20)

# Run the Tkinter event loop
window.mainloop()
