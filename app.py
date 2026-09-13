import numpy as np
import streamlit as st

from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

model = load_model(
    "models/handwritten_character_model.keras"
)

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️"
)

st.title("✍️ Handwritten Digit Recognition")

st.write(
    "Upload a handwritten digit image to predict the digit."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    if st.button("Predict Digit"):

        image = ImageOps.grayscale(image)

        image = image.resize((28, 28))

        image_array = np.array(image)

        if np.mean(image_array) > 127:
            image_array = 255 - image_array

        image_array = image_array.astype("float32") / 255.0

        image_array = image_array.reshape(
            1, 28, 28, 1
        )

        prediction = model.predict(
            image_array
        )[0]

        predicted_digit = np.argmax(
            prediction
        )

        confidence = prediction[
            predicted_digit
        ] * 100

        st.success(
            f"Predicted Digit: {predicted_digit}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )
