
#Face detection from stored images using MTCNN
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os,random
import pdb
import time
import sys
import re
from mtcnn.mtcnn import MTCNN
import argparse
from scipy.optimize import brentq
from scipy import interpolate
from sklearn import metrics

from scipy import misc
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
from scipy import interpolate
import datetime
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputvideo", required=True,
	help="path to input video of faces ")

args = vars(ap.parse_args())





print("[INFO] Detecting faces in training set...")
imagePaths = list(paths.list_images("./datasets/trainimages"))
 

# initialize the total number of faces processed
total = 0

data = []
#MTCNN detector initialised
detector1 = MTCNN()
filepath='./datasets/trainimagesdetected/'


for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	#print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
	dir1 = imagePath.split(os.path.sep)[-2]
	name1 = imagePath.split(os.path.sep)[-1]
	path=filepath+dir1
	filename=path+'/'+name1
	if not os.path.exists(path):
		os.makedirs(path)
	image = cv2.imread(imagePath)
	(h, w) = image.shape[:2]
	result1=detector1.detect_faces(image)
        #Iterate over all of the faces detected and extract their start and end points
        
	for i in result1:
	        bounding_box = i['box']
	        startX=bounding_box[0]
	        endX=bounding_box[0]+bounding_box[3]
	        startY=bounding_box[1]
	        endY=bounding_box[1]+bounding_box[2]
	        confidence1 =i['confidence']
            #if the algorithm is more than 16.5% confident that the detection is a face, show a rectangle around it
	        if (confidence1 > 0.165):
                     face=image[startY-10:endY+10,startX-10:endX+10] 
                     (fH, fW) = face.shape[:2]
 		     # ensure the face width and height are sufficiently large
                     if fW < 20 or fH < 20:
                          continue
                     face=cv2.resize(face, (112, 112))
                     cv2.imwrite(filename,face)
	total += 1
print('Face detection of ' +str(total)+' training images completed')


#Training images filenames and names and are stored 
print('Loading training images labels...')
filepath='./datasets/trainimagesdetected/'
trainimages=[]
trainfolder=[]
imagePaths = list(paths.list_images(filepath))
count=0
for (i, imagePath) in enumerate(imagePaths):

	dir1 = imagePath.split(os.path.sep)[-2]
	name1 = imagePath.split(os.path.sep)[-1]
	dirname1='./datasets/trainimagesdetected/'+dir1
	filename=dirname1+'/'+name1
	trainimages.append(filename)  
	trainfolder.append(dir1)
	count+=1
print('Loading of '+str(count)+ ' training images labels completed')


print('Writing training images labels to text file...')
f= open("./datasets/trainimagesdetected/trainset.txt","w+")
for i in range(len(trainimages)):
    s=trainimages[i]+ ' '+ trainfolder[i]+' '+str(i) +'\n'
    f.write(s)
f.close()
print('Writing completed')



def get_paths( pairs):
    path_list = []
    trainfoldernames=[]
    for pair in pairs:
        path0=pair[0]
        path1=pair[1]

        if os.path.exists(path0) :    # Only add the pair if both paths exist
            path_list.append(path0)
            trainfoldernames.append(path1)
            
    return path_list, trainfoldernames



def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)



image_size='112,112'
image_size = [int(x) for x in image_size.split(',')]
lfw_pairs = read_pairs(os.path.join('./datasets/trainimagesdetected', 'trainset.txt'))



lfw_paths, trainfol = get_paths( lfw_pairs)


i = 0
lfw_bins=[]
for path in lfw_paths:
    with open(path, 'rb') as fin:
        _bin = fin.read()
        lfw_bins.append(_bin)
        i+=1
        if i%10==0:
            print('loading training dataset in bins...', i)



def load_data(db_name, image_size, eval_db_path):

    datasets = np.empty(( (len(lfw_paths)) , image_size[0], image_size[1], 3))
  
    trainfo=trainfol
    for i in range((len(lfw_paths))):
        _bin = lfw_bins[i]   
   
        img = cv2.imdecode(np.frombuffer(_bin, np.uint8), -1)
        img = img - 127.5
        img = img * 0.0078125
        datasets[i, ...] = img
        i += 1

    print(datasets.shape)
    

    
    return datasets, trainfo


def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        #with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    #print('pb model loaded') 
    
    

def trainingset(dataset,imgsize, path, test_batch_size, eval_nrof_folds ):
    with tf.Graph().as_default():
        with tf.Session() as sess:

            ver_list = []
            
            for db in eval_datasets:
                data_set, trainfol1 = load_data(db, image_size, path)       
                ver_list.append(data_set)
             
	
            load_pb(modelname)
  

            # Get input and output tensors, ignore phase_train_placeholder for it have default value.
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            # image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            embedding_size = embeddings.get_shape()[1]
 
            sess.run( tf.global_variables_initializer())
            #print('initialized all global variables')
            for db_index in range(len(ver_list)):
                # Run forward pass to calculate embeddings
                #print('\nRunnning forward pass on {} images')
               
       
                data_sets = ver_list[db_index]
                nrof_batches = data_sets.shape[0] // test_batch_size
                emb_array = np.zeros((data_sets.shape[0], embedding_size))

                for index in range(nrof_batches):
                    start_index = index * test_batch_size
                    end_index = min((index + 1) * test_batch_size, data_sets.shape[0])
                    #print('start_index ' +str(start_index))
                    #print('end index ' +str(end_index))
                    feed_dict = {inputs_placeholder: data_sets[start_index:end_index, ...]}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                
    return emb_array, trainfol1
eval_datasets=['trainset1']
image_size=[112, 112]
# modelname= './MobileFaceNet_TF/MobileFaceNet_TF.ckpt.meta'


modelname='./MobileFaceNet_TF/MobileFaceNet_9925_9680.pb'
eval_db_path= './datasets'
test_batch_size=10
eval_nrof_folds=10
db_name= 'trainset1'
embedd, knownNames= trainingset(eval_datasets,image_size, eval_db_path, test_batch_size, eval_nrof_folds)




data = {"embeddings": embedd, "names": knownNames}
f = open("./datasets/trainimagesdetected/trainsetnew.pkl", "wb")
f.write(pickle.dumps(data))
f.close()





def load_singletestdata(filename):
    test_bins=[]
    image_size=[112,112]
    with open(filename, 'rb') as fin1:
        _bin1 = fin1.read()
        test_bins.append(_bin1)
        
        
    testdataset = np.empty(( 1 , image_size[0], image_size[1], 3))
    _bin = test_bins[0]
       
    img = cv2.imdecode(np.frombuffer(_bin, np.uint8), -1)
    img = img - 127.5
    img = img * 0.0078125
    testdataset[0, ...] = img
     
    return testdataset




def initialise(modelname):
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            load_pb(modelname)
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            embedding_size = embeddings.get_shape()[1]
 
        sess.close()
    print('pb model loaded')        
    return graph,inputs_placeholder, embeddings, embedding_size   



def testsingleimage(testfilename, graph, inputs_placeholder,embeddings, embedding_size ):

        with tf.Session(graph=graph) as sess:
            ver_list = []
            test_data_set = load_singletestdata(testfilename)
            ver_list.append(test_data_set)
            test_batch_size=1
            sess.run(tf.global_variables_initializer())

            for db_index in range(len(ver_list)):

                test_data_sets = ver_list[db_index]
                nrof_batches = test_data_sets.shape[0] // test_batch_size
                emb_array1 = np.zeros((test_data_sets.shape[0], embedding_size))

                for index in range(nrof_batches):
                    start_index = index * test_batch_size
                    end_index = min((index + 1) * test_batch_size, test_data_sets.shape[0])
                    feed_dict = {inputs_placeholder: test_data_sets[start_index:end_index, ...]}
                    emb_array1[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

    
        sess.close()
        return emb_array1


# Face recognition in live stream video using MTCNN

from imutils import paths
import numpy as np

import imutils
import os
from imutils.video import FPS
import time
import random
from mtcnn.mtcnn import MTCNN
import pickle
from imutils.video import VideoStream
from imutils import paths
import cv2
import os


from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# load_pb(model1)
model1='./MobileFaceNet_TF/MobileFaceNet_9925_9680.pb'
graph,inputs_placeholder, embeddings, embedding_size=initialise(model1)


# load the actual face recognition model along with the label encoder
print("[INFO] loading face embeddings...")
data = pickle.loads(open("./datasets/trainimagesdetected/trainsetnew.pkl", "rb").read())
####################################
# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)
# write the actual face recognition model to disk
f = open("./datasets/trainimagesdetected/recognizesetnew", "wb")
f.write(pickle.dumps(recognizer))
f.close()
 
# write the label encoder to disk
f = open("./datasets/trainimagesdetected/leesetnew", "wb")
f.write(pickle.dumps(le))
f.close()


# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("./datasets/trainimagesdetected/recognizesetnew", "rb").read())
le = pickle.loads(open("./datasets/trainimagesdetected/leesetnew", "rb").read())




# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])


print("[INFO] starting video stream...")
_videodir = "./dataset"
#vs = cv2.VideoCapture(0)

dir1='./datasets/videos_t/'
name=args["inputvideo"]
ext='.webm'
stream = cv2.VideoCapture(dir1+name+ext)
fps = FPS().start()



#vs = VideoStream(src=0).start()
writer = None
display=1
count=0
time.sleep(1)
detector = MTCNN()

testfilename='./datasets/testset/test.jpg'
while True:
	
	#frame = vs.read()
	(grabbed,frame) = stream.read()
	if not grabbed:
		break
	count+=1
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	start=time.time()
	(h, w) = image.shape[:2]
	result=detector.detect_faces(image)
        #Iterate over all of the faces detected and extract their start and end points
        
	for i in result:
		bounding_box = i['box']
		confidence1 =i['confidence']
            #if the algorithm is more than 16.5% confident that the      detection is a face, show a rectangle around it
		if (confidence1 > 0.165):
			cv2.rectangle(image, (bounding_box[0], bounding_box[1]),
				(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]), (random.randint(0,255),random.randint(0,255)  , random.randint(0,255)) , 2)
			#end = time.time()
			count = count + 1    
			startX=bounding_box[0]
			endX=bounding_box[0]+bounding_box[3]
			startY=bounding_box[1]
			endY=bounding_box[1]+bounding_box[2]
			face=image[startY-10:endY+10,startX-10:endX+10 ] 
			(fH, fW) = face.shape[:2]
 			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue
			face=cv2.resize(face, (112, 112))
			cv2.imwrite(testfilename,face)	
			start1=time.time()		
			vec=testsingleimage(testfilename,graph,inputs_placeholder, embeddings, embedding_size  )
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = np.round(preds[j]*100,2)
			if proba<30:
				predictedname='unknown'
			else:
				predictedname = le.classes_[j]

			end1=time.time()
			cv2.rectangle(image, (bounding_box[0], bounding_box[1]),
				(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]), (random.randint(0,255),random.randint(0,255)  , random.randint(0,255)) , 2)

			x=random.randint(bounding_box[0],bounding_box[0])
			y=random.randint(bounding_box[1],bounding_box[1]+10)  
			end1=time.time()
			text=predictedname+ '  conf: '+ str(proba)+'Rec'+str(np.round((end1-start1),2))+'Det'+str(np.round((start1-start),2))
			cv2.putText(image, text, (x, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 2)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			end1=time.time()


	if display > 0:
		cv2.imshow("Frame", image)
		key = cv2.waitKey(1) & 0xFF


		# if the `q` key was pressed, break from the loop
		if key == ord("q") :
			#time.sleep(2)
			#cv2.destroyAllWindows()	
			#vs.stop()		
			break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()












