import cv2
import face_recognition
import os
import numpy as np
from ultralytics import YOLO

# ===================== 配置 =====================
FACE_LIB = "face_lib"
VIDEO_PATH = "test3.mp4"
TOLERANCE = 0.5
POSE_CONF = 0.5
LOG_FILE = "student_behavior_log.txt"
# ================================================

# 清空旧日志
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("帧号,学生ID,行为\n")

# 加载人脸库
known_encodings = []
known_ids = []

def load_face_database():
    if not os.path.exists(FACE_LIB):
        os.makedirs(FACE_LIB)
        print("请在 face_lib 里建立每个学生单独文件夹")
        exit()

    for student_id in os.listdir(FACE_LIB):
        stu_dir = os.path.join(FACE_LIB, student_id)
        if not os.path.isdir(stu_dir):
            continue

        # 一个学生多张照片全部加载
        count = 0
        for fname in os.listdir(stu_dir):
            if not fname.endswith(('jpg','png','jpeg')):
                continue
            img_path = os.path.join(stu_dir, fname)
            img = face_recognition.load_image_file(img_path)
            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
                known_ids.append(student_id)
                count += 1
        print(f"✅ {student_id} 加载了 {count} 张人脸")

load_face_database()

# 加载YOLO-Pose
pose_model = YOLO("yolov8n-pose.pt")

# 姿态判行为
def get_behavior(keypoints):
    nose_y = keypoints[0,1] if keypoints[0,2]>0.5 else 9999
    ls_y = keypoints[5,1] if keypoints[5,2]>0.5 else 9999
    rs_y = keypoints[6,1] if keypoints[6,2]>0.5 else 9999
    le_y = keypoints[9,1] if keypoints[9,2]>0.5 else 9999
    re_y = keypoints[10,1] if keypoints[10,2]>0.5 else 9999

    avg_shoulder = (ls_y + rs_y) / 2

    if re_y < avg_shoulder - 15 or le_y < avg_shoulder - 15:
        return "raised hand"
    if nose_y > avg_shoulder - 10:
        return "looking down"
    return "normal posture"

# 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("pose_face_result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. 人脸识别
    face_ids = {}
    face_locs = face_recognition.face_locations(rgb)
    face_encs = face_recognition.face_encodings(rgb, face_locs)

    for (top,right,bottom,left), enc in zip(face_locs, face_encs):
        sid = "unknown"
        if known_encodings:
            dists = face_recognition.face_distance(known_encodings, enc)
            best_idx = np.argmin(dists)
            if dists[best_idx] < TOLERANCE:
                sid = known_ids[best_idx]
        face_ids[(left,top,right,bottom)] = sid
        cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 2)
        cv2.putText(frame, sid, (left,top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # 2. YOLO-Pose + 行为
    pose_res = pose_model(frame, conf=POSE_CONF)
    log_lines = []
    for res in pose_res:
        for box, kp in zip(res.boxes.xyxy, res.keypoints):
            keypoints = kp.data.cpu().numpy().squeeze()
            behav = get_behavior(keypoints)
            x1,y1,x2,y2 = map(int, box)

            # 简单匹配人脸框和人体框，对应学生ID
            current_sid = "unknown"
            for (fl,ft,fr,fb), sid in face_ids.items():
                # 中心近似匹配
                cx1, cy1 = (x1+x2)/2, (y1+y2)/2
                cx2, cy2 = (fl+fr)/2, (ft+fb)/2
                if abs(cx1-cx2) < 80 and abs(cy1-cy2) < 80:
                    current_sid = sid
                    break

            log_lines.append(f"{frame_idx},{current_sid},{behav}")

            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, f"{current_sid}:{behav}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # 写入日志
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    cv2.imshow("FaceID + YOLO-Pose", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ 处理完成！日志已保存到：{LOG_FILE}")
print("日志格式：帧号,学生ID,行为")