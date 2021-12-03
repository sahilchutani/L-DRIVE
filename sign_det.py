import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
 
 
frameWidth= 640         
frameHeight = 480
brightness = 180
threshold = 0.75         
font = cv2.FONT_HERSHEY_SIMPLEX


cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

model = load_model('t2.h5')
 
def grayscale(img):
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		return img
def equalize(img):
		img =cv2.equalizeHist(img)
		return img
def preprocessing(img):
		img = grayscale(img)
		img = equalize(img)
		img = img/255
		return img
def getCalssName(classNo):
		if   classNo == 0: return 'Speed Limit 20 km/h'
		elif classNo == 1: return 'Speed Limit 30 km/h'
		elif classNo == 2: return 'Speed Limit 50 km/h'
		elif classNo == 3: return 'Speed Limit 60 km/h'
		elif classNo == 4: return 'Speed Limit 70 km/h'
		elif classNo == 5: return 'Speed Limit 80 km/h'
		elif classNo == 6: return 'Speed Limit 100 km/h'
		elif classNo == 7: return 'Speed Limit 120 km/h'
		elif classNo == 8: return 'Stop'
		elif classNo == 9: return 'No entry'
		elif classNo == 10: return 'Road work'
		elif classNo == 11: return 'Children crossing'
		elif classNo == 12: return 'Bicycles crossing'
		elif classNo == 13: return 'Turn right ahead'
		elif classNo == 14: return 'Turn left ahead'
		elif classNo == 15: return 'Go straight or right'
		elif classNo == 16: return 'Go straight or left'
		elif classNo == 17: return 'Roundabout mandatory'
 
while True:
		success, imgOrignal = cap.read()
		img = np.asarray(imgOrignal)
		img = cv2.resize(img, (32, 32))
		img = preprocessing(img)
		#cv2.imshow("LDRIVE - Sign Detection", img)
		img = img.reshape(1, 32, 32, 1)
		cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
		cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
		predictions = model.predict(img)
		classIndex = model.predict_classes(img)
		probabilityValue =np.amax(predictions)
		if probabilityValue > threshold:
				cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
				cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
		cv2.imshow("LDRIVE - Sign Detection", imgOrignal)
 

