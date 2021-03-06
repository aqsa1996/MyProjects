import sys
import os
import dlib
import glob
import time
import cv2
start = time.time()


predictor_path = './files/shape_predictor_5_face_landmarks.dat' # Download from http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
face_rec_model_path = './files/dlib_face_recognition_resnet_model_v1.dat' # Download from http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
#faces_folder_path = './frames-my'
output_folder = './clusters'

detector = dlib.get_frontal_face_detector() #a detector to find the faces
sp = dlib.shape_predictor(predictor_path) #shape predictor to find face landmarks
facerec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model

descriptors = []
images = []
vid_path=input('enter video path')

cap= cv2.VideoCapture(vid_path)

while(cap.isOpened()):
  ret, frame = cap.read()
# Load the images from input folder
#for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
  try:
    #print("Processing file: {}".format(frame))
    img = frame

    # Ask the detector to find the bounding boxes of each face. The 1 in the second argument indicates that we should upsample the image 1 time. This will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)

        # Compute the 128D vector that describes the face in img identified by shape.  
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        descriptors.append(face_descriptor)
        images.append((img, shape))
  except:
     continue
  cv2.imshow('frame', frame)
  if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
# Cluster the faces.  
labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
num_classes = len(set(labels)) # Total number of clusters
print("Number of clusters: {}".format(num_classes))

for i in range(0, num_classes):
    indices = []
    class_length = len([label for label in labels if label == i])
    for j, label in enumerate(labels):
        if label == i:
            indices.append(j)
    print("Indices of images in the cluster {0} : {1}".format(str(i),str(indices)))
    print("Size of cluster {0} : {1}".format(str(i),str(class_length)))
    output_folder_path = output_folder +'./'+ str(i) # Output folder for each cluster
    os.path.normpath(output_folder_path)
    os.makedirs(output_folder_path)
    
    # Save each face to the respective cluster folder
    print("Saving faces to output folder...")
    for k, index in enumerate(indices):
        img, shape = images[index]
        file_path = os.path.join(output_folder_path,"face_"+str(k)+"_"+str(i))
        dlib.save_face_chip(img, shape, file_path)
        
print("--- %s seconds ---" % (time.time() - start))
    
    


