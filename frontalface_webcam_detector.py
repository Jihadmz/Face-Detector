import cv2


trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for(x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 6)

    cv2.imshow('pic for testing', frame)
    key = cv2.waitKey(1)

    if key == 113 or key == 81:
        break

print("code completed")
