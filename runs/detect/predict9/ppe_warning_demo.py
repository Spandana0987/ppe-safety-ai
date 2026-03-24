from ultralytics import YOLO
import cv2

model = YOLO("../train2/weights/best.pt")

cap = cv2.VideoCapture("demo_videos/construction_demo1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    labels = [model.names[int(cls)] for cls in results.boxes.cls]

    if "Person" in labels and "Hardhat" not in labels:
        print("⚠ Worker without Hardhat")

    if "Person" in labels and "Safety Vest" not in labels:
        print("⚠ Worker without Safety Vest")

    annotated = results.plot()
    cv2.imshow("PPE Monitor", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
