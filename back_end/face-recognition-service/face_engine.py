"""
独立 CLI 人脸识别签到工具
脱离 Flask 直接运行，使用 OpenCV 窗口显示摄像头画面

用法：
  python face_engine.py
  按 q 退出
"""
import cv2
import face_recognition

from face_db import load_face_database, sign_in
import face_db


def main():
    print("正在加载人脸库...")

    load_face_database()

    if len(face_db.known_encodings) == 0:
        print("❌ 未加载到任何人脸，请检查 faces 文件夹！")
        return

    print(f"加载完成！共 {len(face_db.known_names)} 人")

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
            matches = face_recognition.compare_faces(face_db.known_encodings, encoding)
            name = "未知"

            if True in matches:
                idx = matches.index(True)
                name = face_db.known_names[idx]

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


if __name__ == "__main__":
    main()