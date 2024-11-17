import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

class CarDamageDetector:
    def __init__(self):
        self.model = None
        self.target_size = (224, 224)
        # Define colors for different damage types
        self.colors = {
            'headlight': (0, 255, 255),    # Cyan
            'tail_light': (0, 255, 0),     # Green
            'bumper': (255, 0, 255),       # Magenta
            'door': (0, 165, 255)          # Orange
        }
        try:
            self.load_model()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

    def load_model(self):
        if os.path.exists("Car_detection.model100.h5"):
            self.model = load_model("Car_detection.model100.h5")
            st.success("Model loaded successfully!")
        else:
            st.error("Model file not found!")

    def process_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Create a copy for drawing
        result_image = image.copy()
        height, width = image.shape[:2]

        # Example regions with different colors (replace with actual model predictions)
        regions = [
            {
                'box': [width//4, height//4, 3*width//4, height//2],
                'confidence': 0.92,
                'type': 'bumper'
            },
            {
                'box': [0, height//4, width//4, height//2],
                'confidence': 0.88,
                'type': 'headlight'
            },
            {
                'box': [3*width//4, height//4, width, height//2],
                'confidence': 0.85,
                'type': 'tail_light'
            }
        ]

        # Draw each region
        for region in regions:
            x1, y1, x2, y2 = region['box']
            damage_type = region['type']
            confidence = region['confidence']
            color = self.colors.get(damage_type, (0, 255, 0))

            # Draw rectangle
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Add label with confidence
            label = f"{confidence:.2f}"
            cv2.putText(result_image, label, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result_image, regions

def main():
    st.title("Car Damage Detection")

    # Create sidebar for options
    st.sidebar.title("Options")
    input_option = st.sidebar.radio(
        "Select Input Source:",
        ["Upload Image", "Take Photo"]
    )

    # Initialize detector
    detector = CarDamageDetector()

    if input_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)

            if st.button("Detect Damage"):
                with st.spinner("Processing..."):
                    result_image, damage_info = detector.process_image(image)
                    st.image(result_image, caption="Detected Damage", use_column_width=True)
                    
                    # Display damage analysis
                    st.subheader("Damage Analysis")
                    for i, damage in enumerate(damage_info, 1):
                        st.write(f"Damage {i}:")
                        st.write(f"- Type: {damage['type']}")
                        st.write(f"- Confidence: {damage['confidence']*100:.1f}%")

    else:  # Take Photo
        picture = st.camera_input("Take a picture")
        
        if picture is not None:
            image = Image.open(picture)
            st.image(image, caption="Captured Image", use_column_width=True)
            
            if st.button("Detect Damage"):
                with st.spinner("Processing..."):
                    result_image, damage_info = detector.process_image(image)
                    st.image(result_image, caption="Detected Damage", use_column_width=True)
                    
                    # Display damage analysis
                    st.subheader("Damage Analysis")
                    for i, damage in enumerate(damage_info, 1):
                        st.write(f"Damage {i}:")
                        st.write(f"- Type: {damage['type']}")
                        st.write(f"- Confidence: {damage['confidence']*100:.1f}%")

                    # Add download button
                    if isinstance(result_image, np.ndarray):
                        result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                    else:
                        result_image_pil = result_image
                        
                    buf = io.BytesIO()
                    result_image_pil.save(buf, format="PNG")
                    
                    st.download_button(
                        label="Download Processed Image",
                        data=buf.getvalue(),
                        file_name="damage_detection.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main()