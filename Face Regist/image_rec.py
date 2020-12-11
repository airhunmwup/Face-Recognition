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

    #Loading current image from the image_database
    currrentImage = cv2.imread(f'{image_path}/{kin}')

    #appending all the images to the container variable to hold all images retrieved from database
    imagesContainer.append(currrentImage)

    #Split operation to retrieve just names from the image path to be used for naming matches
    ImageNames.append(os.path.splitext(kin)[0])

#function to get the encodings of images
def face_encodings(image):
    imageDataEncodings = []
    #loop to get every image and encode them
    for img in image:
        #search for the use of cvtColor, to convert images from bgr to rgb before encoding can work
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #encoding the images, results in a list
        encode = fr.face_encodings(img)[0]

        #appending to an empty list
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

#while true mean the program will never stop, infinite loop...so that it keeps runing until something happens
while True:
    #initialize success variable and frame which controls the webcam
    success, frame = video_capture.read()

    #the size display of the video capture
    rgb_frame = frame[:, :, ::-1]

    #this holds the boxes on the indentified face, all this variable are online search them for more details
    face_locations = fr.face_locations(rgb_frame)

    #Given an image, returns the 128-dimension face encoding for each face in the image, the image given is the video web cam
    faceEncodings = fr.face_encodings(rgb_frame, face_locations)

    #top, right, bottom, left implies the box
    for (top, right, bottom, left), faceEncoding in zip(face_locations, faceEncodings):

        #Compare a list of face encodings against a candidate encoding to see if they match.
        matches = fr.compare_faces(knownFaceEncondings, faceEncoding)

        #Given a list of face encodings, compare them to a known face encoding and get a euclidean distance for each comparison face. The distance tells you how similar the faces are.
        #the lower the number the more alike the match is
        faceDistance = fr.face_distance(knownFaceEncondings, faceEncoding)
        
        print(faceDistance)
        #get the minimum number from the face distance of all the known images
        best_match_index = np.argmin(faceDistance)

        name = "Unknown"

        #this part draws the bow and puts the name under
        if matches[best_match_index]:
            name = ImageNames[best_match_index].upper()

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            #attendance function that runs and store the name and time of when the programs identifies someone
            attendanceModel(name)
    
    #stop the never ending while loop when i press q button on the keyboard
    cv2.imshow('Webcam_facerecognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()