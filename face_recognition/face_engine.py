import face_recognition
import cv2
import numpy as np
from datetime import datetime
import os

# ===================== 配置 =====================
SIGN_LOG = "sign_log.txt"
FACE_DIR = "faces/"

# ===================== 加载所有人脸库（已修复！） =====================
def load_face_database():
    known_face_encodings = []
    known_face_names = []
    
    # 遍历文件夹
    for img_name in os.listdir(FACE_DIR):
        if img_name.lower().endswith(("jpg", "png", "jpeg")):
            img_path = os.path.join(FACE_DIR, img_name)
            
            # ✅ 修复点：用 cv2 读取，强制转 RGB，解决格式报错
            img = cv2.imread(img_path)
            if img is None:
                print(f"跳过无效图片：{img_name}")
                continue
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 强制转标准RGB
            
            # 检测是否有人脸
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) == 0:
                print(f"未检测到人脸：{img_name}")
                continue
            
            # 提取特征
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(img_name)[0])
    
    return known_face_encodings, known_face_names

# ===================== 签到记录 =====================
def sign_in(name):
    with open(SIGN_LOG, "a+", encoding="utf-8") as f:
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{name} {time_str}\n")
    print(f"✅ {name} 签到成功！")

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("正在加载人脸库...")
    
    # 自动创建文件夹
    if not os.path.exists(FACE_DIR):
        os.makedirs(FACE_DIR)
        print("已自动创建 faces 文件夹，请放入人脸图片！")
        exit()

    known_encodings, known_names = load_face_database()
    
    if len(known_encodings) == 0:
        print("❌ 未加载到任何人脸，请检查 faces 文件夹！")
        exit()

    print(f"加载完成！共 {len(known_names)} 人")

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    signed_set = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 缩小 + 转RGB
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = small_frame[:, :, ::-1]

        # 检测人脸
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "未知"

            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]

                if name not in signed_set:
                    sign_in(name)
                    signed_set.add(name)

            # 画框
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("人脸识别签到", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()