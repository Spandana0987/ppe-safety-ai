from ultralytics import YOLO
import cv2
from collections import defaultdict, deque

# =========================
# LOAD MODEL
# =========================

model = YOLO("runs/detect/train2/weights/best.pt")

video_path = "demo_videos/construction_demo4.mp4"

cap = cv2.VideoCapture(video_path)

# =========================
# CLASS IDs FROM DATASET
# =========================

HARDHAT = 0
PERSON = 5
SAFETY_VEST = 7
MACHINERY = 8
VEHICLE = 9

# =========================
# TEMPORAL MEMORY SETTINGS
# =========================

MEMORY_FRAMES = 10

helmet_history = defaultdict(lambda: deque(maxlen=MEMORY_FRAMES))
vest_history = defaultdict(lambda: deque(maxlen=MEMORY_FRAMES))

# =========================
# MAIN LOOP
# =========================

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # resize frame to avoid zoom issues
    frame = cv2.resize(frame, (1280,720))

    # =========================
    # YOLO TRACKING MODE
    # =========================

    results = model.track(
        frame,
        persist=True,
        imgsz=960,
        conf=0.18,
        verbose=False
    )[0]

    output = frame.copy()

    persons = []
    helmets = []
    vests = []

    machines = 0
    vehicles = 0

    if results.boxes is None:
        continue

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    ids = results.boxes.id

    if ids is not None:
        ids = ids.cpu().numpy()
    else:
        ids = [None]*len(boxes)

    # =========================
    # CLASSIFY DETECTIONS
    # =========================

    for box, cls, track_id in zip(boxes, classes, ids):

        x1,y1,x2,y2 = map(int,box)
        cls = int(cls)

        if cls == PERSON:

            # ignore tiny distant workers
            if (y2-y1) < 70:
                continue

            persons.append((track_id,x1,y1,x2,y2))

        elif cls == HARDHAT:
            helmets.append((x1,y1,x2,y2))

        elif cls == SAFETY_VEST:
            vests.append((x1,y1,x2,y2))

        elif cls == MACHINERY:
            machines += 1

        elif cls == VEHICLE:
            vehicles += 1


    helmet_violations = 0
    vest_violations = 0

    # =========================
    # CHECK PPE PER WORKER
    # =========================

    for track_id, px1,py1,px2,py2 in persons:

        helmet_found = False
        vest_found = False

        # check helmet overlap
        for hx1,hy1,hx2,hy2 in helmets:
            if hx2>px1 and hx1<px2 and hy2>py1 and hy1<py2:
                helmet_found=True

        # check vest overlap
        for vx1,vy1,vx2,vy2 in vests:
            if vx2>px1 and vx1<px2 and vy2>py1 and vy1<py2:
                vest_found=True

        # =========================
        # UPDATE TEMPORAL MEMORY
        # =========================

        helmet_history[track_id].append(helmet_found)
        vest_history[track_id].append(vest_found)

        helmet_ratio = sum(helmet_history[track_id]) / len(helmet_history[track_id])
        vest_ratio = sum(vest_history[track_id]) / len(vest_history[track_id])

        violation_text = ""

        if helmet_ratio < 0.4:
            violation_text += "No Hardhat "
            helmet_violations += 1

        if vest_ratio < 0.4:
            violation_text += "No Vest"
            vest_violations += 1

        # =========================
        # DRAW VIOLATION
        # =========================

        if violation_text != "":

            cv2.rectangle(output,(px1,py1),(px2,py2),(0,0,255),3)

            cv2.putText(
                output,
                violation_text,
                (px1,py1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

    total_workers = len(persons)

    compliance = 0
    if total_workers > 0:
        compliant = total_workers - (helmet_violations + vest_violations)
        compliance = int((compliant / total_workers) * 100)

    # =========================
    # DASHBOARD PANEL
    # =========================

    cv2.rectangle(output,(10,10),(400,190),(0,0,0),-1)

    cv2.putText(output,f"Workers: {total_workers}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(output,f"Helmet violations: {helmet_violations}",(20,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.putText(output,f"Vest violations: {vest_violations}",(20,100),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.putText(output,f"Machines: {machines}",(20,130),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    cv2.putText(output,f"Vehicles: {vehicles}",(20,160),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    cv2.putText(output,f"Compliance: {compliance}%",(200,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    # =========================
    # SHOW OUTPUT
    # =========================

    cv2.imshow("AI PPE Safety Monitor",output)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
