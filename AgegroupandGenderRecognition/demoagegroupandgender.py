from statistics import mode
import dlib
import cv2
from keras.models import load_model
import numpy as np
#from update_counter import gender
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from update_counter import gender
from update_counter import age
from update_counter import peopele
import time
female_count=0
male_count=0
prev_max=0
kids=0
elders=0
youngadults=0
temp=0
cor=0
diff_cor=0
agegroup_model_path = '../trained_models/agegroup_models/age_group_weights.hdf5'
gender_model_path = '../trained_models/gender_models/updated_weights.hdf5'
agegroup_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
agegroup_offsets = (20, 40)

# loading models
agegroup_classifier = load_model(agegroup_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
agegroup_target_size = agegroup_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
agegroup_window = []


# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture('rtsp://admin:hik@12345@192.168.1.100/1')
counter = 0
start_time = time.time()

while True:

    try:
     bgr_image = video_capture.read()[1]
     gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
     hog_face_detector = dlib.get_frontal_face_detector()
     faces_hog = hog_face_detector(gray_image, 1)
    except:
        continue



    for face_coordinates in faces_hog:


        x = face_coordinates.left()
        y = face_coordinates.top()
        w = face_coordinates.right() - x
        h = face_coordinates.bottom() - y
        face_off=x,y,w,h

        x1, x2, y1, y2 = apply_offsets(face_off, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]
        x1, x2, y1, y2 = apply_offsets(face_off, agegroup_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

       # x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        #gray_face = gray_image[y1:y2, x1:x2]

        try:
         rgb_face = cv2.resize(rgb_face, (gender_target_size))
         gray_face = cv2.resize(gray_face, (agegroup_target_size))

        except:
           continue

        rgb_face = np.expand_dims(rgb_face, 2)
        rgb_face = np.expand_dims(rgb_face, 0)

        rgb_face = preprocess_input(rgb_face, False)
        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        #cross corrletion
        try:
         if(temp>0):
          arr1_1D=np.reshape(cor, (len(cor)*4096))

          arr2_1D=np.reshape(rgb_face, (len(rgb_face)*4096))


          cor_out=np.correlate(arr1_1D,arr2_1D,'full')
         #print(cor)
          maxcor=np.max(cor_out)
          diff_cor=np.abs(prev_max-maxcor)
        # print(cor_out)
          print('difference',diff_cor)
          print ('maximum',maxcor)
          prev_max=maxcor
        except:
         continue
        cor=rgb_face
        temp+=1


        try:
         gender_prediction = gender_classifier.predict(rgb_face)
         gender_label_arg = np.argmax(gender_prediction)
         gender_text = gender_labels[gender_label_arg]

         agegroup_label_arg = np.argmax(agegroup_classifier.predict(gray_face))
         agegroup_text = agegroup_labels[agegroup_label_arg]
         agegroup_window.append(agegroup_text)
         counter += 1
         print("FPS: ", counter / (time.time() - start_time))
         print('no of frames',counter)
         print('time',time.time() - start_time)



         if gender_text == gender_labels[0]:
         #   color = (0, 0, 255)
            if(diff_cor>150):
             female_count+=1

         else:
           # color = (255, 0, 0)
            if(diff_cor>150):
             male_count+=1
         total = [male_count, female_count]
         #gender(total)
         if agegroup_text == agegroup_labels[0]:
            if(diff_cor>150):
             kids+=1

         elif agegroup_text == agegroup_labels[1]:
            if(diff_cor>150):
             elders+=1
         else:
            if(diff_cor>150):
             youngadults+=1
         totalage = [youngadults, elders, kids]
         #age(totalage)
         peoplecount=male_count+female_count
         #peopele(peoplecount,peoplecount,peoplecount)


        except:
         continue
        print('female', female_count)
        print('male', male_count)
        print('kids', kids)
        print('young adults', youngadults)
        print('elders', elders)


    cv2.imshow('window_frame', rgb_image)
    #start_time = time.time()

   # if (time.time() - start_time) > x:

    if cv2.waitKey(20) & 0xFF == ord('q'):
     break


print('total female',female_count)
print('total male',male_count)
print('total kids',kids)
print('total young adults',youngadults)
print('total elders',elders)

