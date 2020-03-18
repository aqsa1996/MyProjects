from model import create_model
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from align import AlignDlib
from sklearn.metrics import f1_score, accuracy_score


from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')



class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

metadata = load_metadata('images')




#matplotlib inline

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

# Load an image
orig = load_image('testimg.jpg')

# Detect face and return bounding box
bb = alignment.getLargestFaceBoundingBox(orig)

# Transform image using specified face landmark indices and crop image to 96x96
aligned = alignment.align(96, orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)



def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


embedded = np.zeros((metadata.shape[0], 128))
test_emb=np.zeros((metadata.shape[0], 128))
for i, m in enumerate(metadata):
    #print (m.image_path())
    #img = load_image(m)
    try:
     img=cv2.imread(m.image_path())
     img = align_image(img)
     #print ('img',img)
     # scale RGB values to interval [0,1]
     img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
     embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
    except:
        continue

print ('embedded',embedded[1])
aligned = (aligned / 255.).astype(np.float32)
test_emb = nn4_small2_pretrained.predict(np.expand_dims(aligned, axis=0))[0]
print(test_emb)


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair( idx2):
    a=distance(test_emb,embedded[idx2])
    cv2.imshow('img1',aligned)
    cv2.imshow('img2',load_image(metadata[idx2].image_path()))
    cv2.waitKey(0)
    print (a)

#show_pair( 120)
#show_pair( 1500)


targets = np.array([m.name for m in metadata])

encoder = LabelEncoder()
encoder.fit(targets)

# Numerical encoding of identities
y = encoder.transform(targets)

train_idx = np.arange(metadata.shape[0]) % 2 != 0
test_idx = np.arange(metadata.shape[0]) % 2 == 0

# 50 train examples of 10 identities (5 examples each)
X_train = embedded[train_idx]
# 50 test examples of 10 identities (5 examples each)
X_test = embedded[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
svc = LinearSVC()

knn.fit(X_train, y_train)
svc.fit(X_train, y_train)

acc_knn = accuracy_score(y_test, knn.predict(X_test))
acc_svc = accuracy_score(y_test, svc.predict(X_test))

print('KNN accuracy', acc_knn, 'SVM accuracy',  acc_svc)



example_image = aligned
example_prediction = svc.predict()
example_identity = encoder.inverse_transform(example_prediction)[0]

cv2.imshow('test',example_image)
cv2.waitKey(0)
#plt.title(f'Recognized as {example_identity}')
print('recog as:',example_identity)
