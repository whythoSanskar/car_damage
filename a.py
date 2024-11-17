import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("Car_detection.model100.h5")

# Function to preprocess a frame for the model
def preprocess_frame(frame, target_size=(224, 224)):
    original_size = frame.shape[:2]  # Save original dimensions (height, width)
    frame = cv2.resize(frame, target_size)  # Resize to target size
    frame = frame / 255.0  # Normalize to [0, 1]
    return np.expand_dims(frame, axis=0), original_size

# Function to post-process the prediction
def postprocess_prediction(prediction, original_size):
    prediction = prediction.squeeze()  # Remove batch and channel dimensions
    prediction = cv2.resize(prediction, original_size[::-1])  # Resize to original size
    return prediction

# Function to draw bounding boxes and highlight damage
def draw_detections(frame, mask):
    binary_mask = (mask > 0.5).astype(np.uint8)  # Convert to binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a transparent overlay for damage highlighting
    damage_overlay = frame.copy()
    
    for contour in contours:
        # Ignore small contours (noise)
        if cv2.contourArea(contour) < 500:
            continue
        
        # Draw filled contour on overlay (red highlighting for damage)
        cv2.drawContours(damage_overlay, [contour], -1, (0, 0, 255), -1)
        
        # Draw bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
        
        # Calculate confidence score
        box_mask = mask[y:y + h, x:x + w]
        confidence = np.mean(box_mask) * 100
        
        # Display confidence score
        cv2.putText(frame, f"Damage: {confidence:.2f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Blend the overlay with original frame
    alpha = 0.3  # Transparency factor
    result_frame = cv2.addWeighted(frame, 1, damage_overlay, alpha, 0)
    
    return result_frame

# Main function for live detection
def live_detection():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to quit.")
    print("Press 's' to save the current frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # Preprocess the frame
        preprocessed_frame, original_size = preprocess_frame(frame)

        # Predict the damage mask
        prediction = model.predict(preprocessed_frame)

        # Post-process the prediction
        mask = postprocess_prediction(prediction, original_size)

        # Draw detections and highlights
        result_frame = draw_detections(frame, mask)

        # Display the results
        cv2.imshow("Car Damage Detection", result_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"damage_detection_{timestamp}.jpg", result_frame)
            print(f"Frame saved as damage_detection_{timestamp}.jpg")

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the live detection
if __name__ == "__main__":
    live_detection()