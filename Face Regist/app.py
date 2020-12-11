from typing import NamedTuple
import numpy as np
import face_recognition as fr
import cv2
import os
from datetime import datetime


# Varible to hold the image database path
image_path = 'assets/images'

# image_database recieves a list of images from the database path
image_database = os.listdir(image_path)

# set to enable video stream from a webcam
# since we will be using a webcam to capture matches
video_capture = cv2.VideoCapture(0)

# Container to hold all images retrieved from database
imagesContainer = []

# Variable to hold names of all images in Database
ImageNames = []
print(image_database)


for kin in image_database:
    currrentImage = cv2.imread(f'{image_path}/{kin}')
    imagesContainer.append(currrentImage)
    ImageNames.append(os.path.splitext(kin)[0])


def face_encodings(image):
    imageDataEncodings = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        imageDataEncodings.append(encode)
    return imageDataEncodings


knownFaceEncondings = face_encodings(imagesContainer)
print(len(knownFaceEncondings))

def attendanceModel(name):
    with open('saves/attendance.csv', 'r+') as f:
        attendanceData = f.readlines()
        namesDataList = []
        for line in attendanceData:
            entry = line.split(',')
            namesDataList.append(entry[0])
        if name not in namesDataList:
            now = datetime.now()
            dateString = now.strftime('%H:%M')
            f.write(f'\n{name},{dateString}')


while True:
    success, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    faceEncodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), faceEncoding in zip(face_locations, faceEncodings):
        matches = fr.compare_faces(knownFaceEncondings, faceEncoding)
        faceDistance = fr.face_distance(knownFaceEncondings, faceEncoding)
        
        print(faceDistance)

        best_match_index = np.argmin(faceDistance)

        if matches[best_match_index]:
            name = ImageNames[best_match_index].upper()

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            attendanceModel(name)
    
    cv2.imshow('Webcam_facerecognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()