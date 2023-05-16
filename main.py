import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture=cv2.VideoCapture(0)
# load know face
sourabh_image=face_recognition.load_image_file("faces/sourabh.jpg")
sourabh_encoding=face_recognition.face_encodings(sourabh_image)[0]
p2_image=face_recognition.load_image_file("faces/p2.jpg")
p2_encoding=face_recognition.face_encodings(p2_image)[0]

known_face_encodings=[sourabh_encoding,p2_encoding]
known_face_names=['sourabh','p2']

employes=known_face_names.copy()

face_locations=[]
face_encodings=[]

now=datetime.now()
current_date=now.strftime("%y-%m-%d")

f=open(f"{current_date}.csv","w+",newline="")
lnwriter = csv.writer(f)

while True:
    _, frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,00),fx=0.25,fy=0.25)
    rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    #FACE
    face_locations=face_recognition.face_Locations(rgb_small_frame)
    face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)

    for face_encodings in face_encodings:
        matches=face_recognition.compare_faces(known_face_encodings,face_encodings)
        face_distance=face_recognition.face_distance(known_face_encodings,face_encodings)
        best_match_index=np.argmin(face_distance)


        if(matches[best_match_index]):
            name= known_face_names[best_match_index]

    #add the text
        if name in known_face_names:
            font=cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText=(10,100)
            fontScale=1.5
            fontColor=(255,0,0)
            thickness=3
            lineType = 2
            cv2.putText(frame,name+"present",bottomLeftCornerOfText,font,fontColor,fontScale,thickness.lineType)
        if name in employes:
            employes.remove(name)
            current_time=now.strftime("%H-%M%S")
            lnwriter.writerow(([name,current_time]))
    cv2.imshow("attendence",frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()