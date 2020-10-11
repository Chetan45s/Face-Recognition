import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './dataset/'

filename = str(input("Enter the name : "))

while True:
    ret,frame = cap.read()
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces1 = face_cascade.detectMultiScale(frame,1.3,5)   # This returns the tuple of coordinates of rectange containing face.

    faces = sorted(faces1,key = lambda k: k[2]*k[3])

    
    for (x,y,w,h) in faces[-1:]:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # uptill we have images coordinates now we want to extract the largest image from the set of images and want to select region of
        # intrest like face.

        # we also want padding of 10px on each side so 
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset] # setting the coordinates for padding
        face_section = cv.resize(face_section,(100,100))   # resizing the image upto 100x100px
        skip += 1
        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))

    cv.imshow("Video Frame",frame)
    cv.imshow("Face Section",face_section)

    # wait for user input- q, then you will stop the loop

    key_pressed = cv.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+filename+'.npy',face_data)
print("Data Successfully saved at ",dataset_path+filename+'.npy')

cap.release()
cv.destroyAllWindows()