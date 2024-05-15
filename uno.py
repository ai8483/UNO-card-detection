from ultralytics import YOLO
import cv2
import cvzone
import math

# Attempt to initialize the webcam with default index 0
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Load the YOLO model
model = YOLO("uno.pt")

# Define class names
classNames = ['9', '1', '10', 'plus card', 'reverse card', 'wild plus card', 'wild card', '2', '3', '4', '5', '6', '7', '8', '0']

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image")
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5:
                myColor = (255, 0, 0)  # Default color for bounding box
                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1.3, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=8)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 2)  # Drawing bounding box with the specified color

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
