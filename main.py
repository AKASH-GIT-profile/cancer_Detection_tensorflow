# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

def main():
    st.title("Image Upload and Display App")
    model = tf.keras.models.load_model('odette.h5')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","bmp"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Convert the uploaded image to a numpy array
        img = Image.open(uploaded_file)
        img = img.resize((224, 224))  # Adjust size based on your model's input size
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        value = int(pred > 0.5)
        st.write("Classifying...")
        if value==0:
            st.success("Given Image is a Cancer Infected Cell")
        if value==1:
            st.success("Given Image is a Normal Cell")



        # Perform any additional processing or analysis here
        # For example, you can use a pre-trained model for image classification.

        # Display the result or any other output
        # st.write("Result:", result)

if __name__ == "__main__":
    main()
