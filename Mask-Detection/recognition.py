import face_recognition
import cv2
import numpy as np


cam = cv2.VideoCapture(0)
#loading of dataset
#Samuel
samuel_image = face_recognition.load_image_file("samuel.jpg")
samuel_face_encoding = face_recognition.face_encodings(samuel_image)[0]

#Fred

fred_image = face_recognition.load_image_file("fred.jpg")
fred_face_encoding = face_recognition.face_encodings(fred_image)[0]

#Dickson

dickson_image = face_recognition.load_image_file("dickson.jpg")
dickson_face_encoding = face_recognition.face_encodings(dickson_image)[0]

#Owen

owen_image = face_recognition.load_image_file("owen.jpg")
owen_face_encoding = face_recognition.face_encodings(owen_image)[0]

#aboubakar

aboubakar_image = face_recognition.load_image_file("abubakar.jpg")
aboubakar_face_encoding = face_recognition.face_encodings(aboubakar_image)[0]

#suleiman
suleiman_image = face_recognition.load_image_file("suleiman.jpg")
suleiman_face_encoding = face_recognition.face_encodings(suleiman_image)[0]

#Bongani
bongani_image = face_recognition.load_image_file("bongani.jpg")
bongani_face_encoding = face_recognition.face_encodings(bongani_image)[0]

 
#....fill...


known_face_encodings = [
   samuel_face_encoding,
     fred_face_encoding ,
    dickson_face_encoding,
    owen_face_encoding ,
    aboubakar_face_encoding ,
    suleiman_face_encoding ,
   bongani_face_encoding 
   
   #...fill...
  
]
known_face_names = [
    "Samuel",
    "Fred" ,
    "Dickson",
    "Owen" ,
    "Aboubakar" ,
    "Suleiman" ,
    "Bongani"
   
    #...fill...
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
   
    ret, frame = cam.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
    
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
           
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"   
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]


            face_names.append(name)

    process_this_frame = not process_this_frame
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4 
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Video', frame) 
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows()
cam.stop()

