import face_recognition as fr
import imutils
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
import PIL
from PIL import Image
from mtcnn.mtcnn import MTCNN

from imutils import paths
import time
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

def facedetection(imagePath):
        im = PIL.Image.open(imagePath)
        mode='RGB'
    
        im = im.convert(mode)
        image=np.array(im)

        (h, w) = image.shape[:2]
        detector1=MTCNN()
        result1=detector1.detect_faces(image)
        facebox=[]
        for i in result1:
                bounding_box = i['box']
                startX=bounding_box[0]
                endX=bounding_box[0]+bounding_box[3]
                startY=bounding_box[1]
                endY=bounding_box[1]+bounding_box[2]
                facebox.append(startX)
                facebox.append(endX)
                facebox.append(startY)
                facebox.append(endY)

        return result1


def compare_faces(file1, file2):
    # Load the jpg files into numpy arrays
    image1 = fr.load_image_file(file1)
    image2 = fr.load_image_file(file2)
    
    # Get the face encodings for 1st face in each image file
    image1_encoding = fr.face_encodings(image1)[0]
    image2_encoding = fr.face_encodings(image2)[0]
    
    # Compare faces and return True / False
    results = fr.compare_faces([image1_encoding], image2_encoding)    
    return results[0]     


# Each face is tuple of (Name,sample image)    
known_faces = [('tamil','datasets/trainimages/tamil/6.jpg'),
               ('om','datasets/trainimages/om/1.jpg'),
               ('rithvik','datasets/trainimages/rithvik/3.jpg'),
              ]
    
def face_rec(file):
    """
    Return name for a known face, otherwise return 'Uknown'.
    """
    for name, known_file in known_faces:
        if compare_faces(known_file,file):
            return name
    return 'Unknown' 

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

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        #with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    #print('pb model loaded') 
def training_images():
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
	f= open("./datasets/trainimagesdetected/trainset1.txt","w+")
	for i in range(len(trainimages)):
	    s=trainimages[i]+ ' '+ trainfolder[i]+' '+str(i) +'\n'
	    f.write(s)
	f.close()
	print('Writing completed')
	image_size='112,112'
	image_size = [int(x) for x in image_size.split(',')]
	lfw_pairs = read_pairs(os.path.join('./datasets/trainimagesdetected', 'trainset1.txt'))
	lfw_paths, trainfol = get_paths( lfw_pairs)


	ki = 0
	lfw_bins=[]
	for path in lfw_paths:
	    with open(path, 'rb') as fin:
	        _bin = fin.read()
	        lfw_bins.append(_bin)
	        ki+=1
	        if ki%10==0:
	            print('loading training dataset in bins...', ki)
	image_size=[112, 112]
	datasets = np.empty(( (len(lfw_paths)) , image_size[0], image_size[1], 3))
  
   
	for i in range((len(lfw_paths))):
		_bin = lfw_bins[i]   
   
		img = cv2.imdecode(np.frombuffer(_bin, np.uint8), -1)
		img = img - 127.5
		img = img * 0.0078125
		datasets[i, ...] = img
		i += 1
	print(datasets.shape)  
    
	
	modelname='./MobileFaceNet_TF/MobileFaceNet_9925_9680.pb'	
	test_batch_size=10	
	embedd, knownNames= trainingset(datasets, trainfol, test_batch_size,modelname)
	data = {"embeddings": embedd, "names": knownNames}
	f = open("./datasets/trainimagesdetected/trainset.pkl", "wb")
	f.write(pickle.dumps(data))
	f.close()
	# load the actual face recognition model along with the label encoder
	print("[INFO] loading face embeddings...")
	data = pickle.loads(open("./datasets/trainimagesdetected/trainset.pkl", "rb").read())
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
	f = open("./datasets/trainimagesdetected/recognizeset", "wb")
	f.write(pickle.dumps(recognizer))
	f.close()
 
	# write the label encoder to disk
	f = open("./datasets/trainimagesdetected/leeset", "wb")
	f.write(pickle.dumps(le))
	f.close()

	return True






    

def trainingset(data_set, trainfol1, test_batch_size,modelname ):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_pb(modelname)
            ver_list = []
            
                 
            ver_list.append(data_set)
  

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




     
        

def initialise(modelname):
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:

            load_pb(modelname)
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            embedding_size = embeddings.get_shape()[1]
 
        sess.close()
    print('model initialised')        
    return graph,inputs_placeholder, embeddings, embedding_size   

def load_singletestdata(img):
    
    image_size=[112,112]    
        
    testdataset = np.empty(( 1 , image_size[0], image_size[1], 3))
    img = img - 127.5
    img = img * 0.0078125
    testdataset[0, ...] = img
     
    return testdataset

def testimage(data_set, graph, inputs_placeholder,embeddings, embedding_size ):

        with tf.Session(graph=graph) as sess:
            ver_list=[]           
            test_batch_size=4
            sess.run(tf.global_variables_initializer())
             
            
            ver_list.append(data_set)
            for db_index in range(len(ver_list)):
                test_data_sets = ver_list[db_index]
                nrof_batches = test_data_sets.shape[0] // test_batch_size
                emb_array1 = np.zeros((test_data_sets.shape[0], embedding_size))

                for index in range(nrof_batches):
                    start_index = index * test_batch_size
                    end_index = min((index + 1) * test_batch_size, test_data_sets.shape[0])
                    print(start_index)
                    print(end_index)
                    feed_dict = {inputs_placeholder: test_data_sets[start_index:end_index, ...]}
                    emb_array1[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

    
        sess.close()
        return emb_array1



def load_face_data(imagePath):
	start=time.time()
	detector = MTCNN()
	im = PIL.Image.open(imagePath)
	mode='RGB'
	im = im.convert(mode)
	image=np.array(im)
	(h, w) = image.shape[:2]	
	result=detector.detect_faces(image)
	end=time.time()
        #Iterate over all of the faces detected and extract their start and end points
	resp_data=[]    
	count=1  
	faceid=[]
	ii=0  
	datasets = np.empty(((len(result) ), 112, 112,3))
	for i in result:
		bounding_box = i['box']

		confidence1 =i['confidence']
            #if the algorithm is more than 16.5% confident that the      detection is a face, show a rectangle around it
		if (confidence1 > 0.165):
			ret=[]
			  #save the modified image to the Output folder
			startX=bounding_box[0]
			endX=bounding_box[0]+bounding_box[3]
			startY=bounding_box[1]
			endY=bounding_box[1]+bounding_box[2]

			face=image[startY-10:endY+10,startX-10:endX+10 ] 
			(fH, fW) = face.shape[:2]
 			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue
			ret.append(startX)
			ret.append(bounding_box[0]+bounding_box[2])
			ret.append(startY)
			ret.append(bounding_box[1]+bounding_box[3])
			img=cv2.resize(face, (112, 112))
			img = img - 127.5
			img = img * 0.0078125
			datasets[ii, ...] = img
			resp_data.append(ret)
			faceid.append(count)
			count+=1
			ii += 1
	print(datasets.shape)
	detectiontime=round((end-start),3)
	return datasets, resp_data,faceid,detectiontime


def face_rec_initialise(imagePath):
	if training_images() is True:
		print('Training is completed')
	st=time.time()
	facedata={}
	datanew=[]
	# load_pb(model1)
	model1='./MobileFaceNet_TF/MobileFaceNet_9925_9680.pb'
	graph,inputs_placeholder, embeddings, embedding_size=initialise(model1)


	et=time.time()

	start=time.time()
	testdataset, box, faceid1, detection_time= load_face_data(imagePath)  
	end=time.time()   
	start1=time.time()
	vec=testimage(testdataset,graph,inputs_placeholder, embeddings, embedding_size  )
	# load the actual face recognition model along with the label encoder
	#data = pickle.loads(open("./datasets/trainimagesdetected/trainsetnew.pkl", "rb").read())
	recognizer = pickle.loads(open("./datasets/trainimagesdetected/recognizeset", "rb").read())
	le = pickle.loads(open("./datasets/trainimagesdetected/leeset", "rb").read())

	# encode the labels
	#print("[INFO] encoding labels...")
	#le = LabelEncoder()
	#labels = le.fit_transform(data["names"])

	predictedname=[]
	proba=[]
	preds=[]
	for k in range(len(vec)):
		preds = recognizer.predict_proba(vec)[k]
		j = np.argmax(preds)
		proba1 = np.round(preds[j]*100,2)
		if proba1<10:
			predictedname1='unknown'
		else:
			predictedname1 = le.classes_[j]
		proba.append(proba1)
		predictedname.append(predictedname1)
	end1=time.time()
	inittime=round((et-st),3)
	loadtime=round((round((end-start),3)-detection_time),3)
	rectime=round((end1-start1),3)
	#print(predictedname)
	#print(faceid1)
	facedata=[]
	record={}
	keys=["PredictedName","Confidence", "face_id","startX","endX","startY","endY"]
	
	for (p,c,f, b) in zip(predictedname,proba,faceid1,box): 
		record={keys[0]:p,keys[1]:c,keys[2]:f,keys[3]:int(b[0]),keys[4]:int(b[1]),keys[5]:int(b[2]),keys[6]:int(b[3])}
		facedata.append(record)
	record={"inittime":inittime,"preprocessing":loadtime,"recognition":rectime,"detectiontime":detection_time}
	facedata.append(record)
	#print(facedata)
	return facedata
