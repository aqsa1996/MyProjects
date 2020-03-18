import os
import glob
import cv2
import csv
path = './clusters/'
import os

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
       if (file==os.listdir(r)[0]):
        files.append(os.path.join(r, file))

for bb,file in enumerate (files):
    print(bb,file)
    a= cv2.imread(file)
    #cv2.imwrite('./WebApp/static/images/img{}.png'.format(bb), a)
    cv2.imwrite('./images/img{}.png'.format(bb), a)
    #wait for 1 second
    k = cv2.waitKey(1000)
    #destroy the window
    cv2.destroyAllWindows()


path1 = './images/'
i=0
with open('./labels.csv', 'w', newline='') as csvfile:
 writer = csv.writer(csvfile, delimiter=' ',

                           quoting=csv.QUOTE_MINIMAL)


 writer.writerow(['cluster no',',','label'])
 for f in glob.glob(os.path.join(path1, "*.png")):
   img=cv2.imread(f)
   cv2.imshow('person',img)
   cv2.waitKey(0)
   name=input('Enter name')
   key=name
   writer.writerow([i,',',key])
   i+=1



