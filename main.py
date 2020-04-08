import time
import cascade_classifier as cc

import cv2


def detect_by_nose_and_mouth(face_image):
    """
    :param face_image:numpy array,  the image that contain faces
    :return is_wearing_mask:int {
        -1: not wearing,
        0: wearing but not fit, ( TODO
        1: wearing correctly
    }
    """
    flag_mouth = flag_nose = False
    noses = cc.nose.detectMultiScale(face_image)
    if noses is not None and len(noses) > 0:
        flag_nose = True

    if flag_nose:
        return -1

    else:
        return 1


def get_faces(gray_image):
    faces = cc.face.detectMultiScale(gray_image)
    for x, y, w, h in faces:
        yield x, y, w, h, gray_image[y:y + h, x:x + w]


warning_lasting_count = 0


def process_frame(frame):
    global warning_lasting_count
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for x, y, w, h, face in get_faces(frame_gray):
        is_wearing = detect_by_nose_and_mouth(face)
        if is_wearing != 1:
            warning_lasting_count += 1
        else:
            warning_lasting_count = 0

        if warning_lasting_count > 5:
            # print("Oops! someone is exposed to danger of China Virus!")
            frame = cv2.putText(frame,
                                "Oops! someone is exposed to danger of China Virus!",
                                (30 + 4, frame_gray.shape[0] // 2 + 10 + 3),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, .9, (0, 0, 0), 2)

            frame = cv2.putText(frame,
                                "Oops! someone is exposed to danger of China Virus!",
                                (30, frame_gray.shape[0] // 2 + 10),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, .9, (0, 0, 255), 2)

        color = (0, 255, 0) if is_wearing == 1 else ((0, 255, 255) if is_wearing == 0 else (0, 0, 255))  # BGR
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return frame


def main():
    capture = cv2.VideoCapture(0)
    last_frame_time = time.time()
    while True:
        current_time = time.time()
        time_spent = current_time - last_frame_time + 1e-5
        fps = 1 / time_spent
        print("FPS:", fps)
        last_frame_time = current_time
        _, frame = capture.read()
        frame = process_frame(frame)
        frame = cv2.putText(frame, f"{int(fps)}", (10, 20), cv2.FONT_HERSHEY_DUPLEX, .6, (0, 255, 0))
        cv2.imshow("win", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == '__main__':
    main()
