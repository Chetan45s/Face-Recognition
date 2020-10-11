import cv2 as cv
import numpy as np
import os

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

def distance(a,b):
    return np.sqrt(sum((a-b)**2))

def knn(train,test):
    k = 5
    dist = []

    for i in range(train.shape[0]):
        ix = train[i,:-1]
        iy = train[i, -1]

        d = distance(test,ix)
        dist.append([d,iy])
    
    # sort based on distance and get top k
    dk = sorted(dist,key=lambda x: x[0])[:k]

    #Retrieve only the labels
    labels = np.array(dk)[:,-1]

    #Get frequencies of each label
    output = np.unique(labels,return_counts=True)

    #Find max frequenct and corresponding label
    index = np.argmax(output[1])
    return output[0][index]
        

skip = 0
dataset_path = './dataset/'
face_data = []
label = []

class_id = 0  # label for the given file
names = {} #Mapping between id - name

# Data preparation

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        #create a mapping bet name_id and name
        names[class_id] = fx[:-4]
        print("Loaded : "+fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)


        #create labels for the class of same image

        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1

        # we have two data ser target contain name_id of the face and data_item contain data of the image
        label.append(target)


# concatinating data
face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(label,axis=0).reshape((-1,1))

# print(face_dataset.shape)
# print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)


# Training

while True:
    ret,frame = cap.read()
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)   # This returns the tuple of coordinates of rectange containing face.

    # faces = sorted(faces,key = lambda k: k[2]*k[3])

    
    for face in faces:
        # cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # uptill we have images coordinates now we want to extract the largest image from the set of images and want to select region of
        # intrest like face.
        x,y,w,h = face
        # we also want padding of 10px on each side so 
        offset = 10
        face_section1 = frame[y-offset:y+h+offset,x-offset:x+w+offset] # setting the coordinates for padding
        face_section = cv.resize(face_section1,(100,100))   # resizing the image upto 100x100px

        #predicted Label (out)
        out = knn(trainset,face_section.flatten())

        # Display on the screen the name and rectange around it
        pred_name = names[int(out)]
        cv.putText(frame,pred_name,(x,y-10),cv.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,0,0),2,cv.LINE_AA)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    
    cv.imshow("Faces",frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()