"""Requirements:  Tensorflow, Numpy, Open cv which is imported as cv2, imutils et keras. We are using the mobilenet_v2 architecture.  
We import the os which will follow the path to our dataset """
import tensorflow as tf
import numpy as np
from numpy import array
import os 
import cv2
import imutils as imutils
from tensorflow import keras
from imutils import resize
from  imutils.video import VideoStream
from cv2 import putText, imshow, FONT_HERSHEY_DUPLEX, rectangle, destroyAllWindows
from  cv2 import cvtColor, resize, COLOR_BGR2RGB, VideoCapture, waitKey
from cv2.dnn import blobFromImage, readNet


"""Cette fonction def nous assure de mettre la main sur les dimensions nécessaires, nous utilisons ensuite les dimensions pour construire un gret objet binaire initié en tant que blpob
Afin d’obtenir la détection faciale;- Nous passons un gros objet binaire à travers le réseau CNN.
Cela nous permettra de boucler les détections. Nous allons extraire la probabilité de la détection et l’exprimer sous forme de confiance
nous ajoutons les cases délimitant les visages puis chargeons notre modèle à partir de disq"""

parameters = os.path.join(r"parameters.prototxt")	
visage_model = os.path.join(r"PIL.caffemodel")

print_index = []


def visage_et_masque_detecté_predecté(hog,visage, tempsreel_masque):                                                                                                                                           
	(hog_L, hog_B) = hog.shape[:2]



	visage.setInput(blobFromImage(hog, 1.0, (256, 256),(104.0, 177.0, 123.0)))
	perceives= visage.forward()
	print(perceives.shape)  
    
	visage_detecté = []
	visage_location = []
	prédections = []
     
	for vec_3 in range(0, perceives.shape[2]): 
		sure = perceives[0, 0, vec_3, 2]
		if sure > 0.50:
			rect = perceives[0, 0, vec_3, 3:7] * np.array([hog_B, hog_L, hog_B, hog_L])
			print (rect)
			(startA, startB, endA, endB) = rect.astype("int")

			(startA, startB) = (max(0, startA), max(0, startB))
			(endA, endB) = (min(hog_B - 1, endA), min(hog_L - 1, endB))
			detecté_fc = hog[startB:endB, startA:endA]
			detecté_fc = cvtColor(detecté_fc, COLOR_BGR2RGB)
			detecté_fc = resize(detecté_fc, (256, 256))
			detecté_fc = keras.preprocessing.image.img_to_array(detecté_fc)
			detecté_fc = keras.applications.mobilenet_v2.preprocess_input(detecté_fc)
			visage_detecté.append(detecté_fc)
			visage_location.append((startA, startB, endA, endB))         
	if len(visage_detecté) > 0:
		visage_detecté = array(visage_detecté, dtype="float32")
		prédections = tempsreel_masque.predict(visage_detecté, batch_size=52)
       
		
	return (visage_location, prédections, print_index)  


print ("Preparing  indexes between 0 et 1")
print_index.append(parameters)


print("Opening Camera")



visage = readNet ( parameters, visage_model )
tempsreel_masque = keras.models.load_model("maskedfacebrk_100epoche.model")


"""Démarrage de notre diffusion en direct: nous avons utilisé une webcam PC à caméra interne et une caméra externe de téléphone Samsung dont l’adresse IP est http://192.168.43.1:8080/video
Le flux en direct nous permet de passer en boucle sur les images
détection du visage: Nous donnons de la couleur à nos boîtes englobantes et incluons la probabilité sur le cadre, affichant le cadre final, nettoyage « " »"""
            
cam = VideoStream(src=0).start()
VideoCapture(0)     
    
while True:
	hog =cam.read()
	hog = imutils.resize(hog,width=500)
	(visage_location, predetectéions, print_index) = visage_et_masque_detecté_predecté(hog, visage, tempsreel_masque)

	for (rect, spae) in zip(visage_location, predetectéions):

		(startA, startB, endA, endB) = rect
		(masque, sansmasque) = spae

		tag = "masque" if masque > sansmasque else "No masque"
		tint = (0, 255, 0) if tag == "masque" else (0, 0, 255)

		tag = "{}: {:.2f}%".format(tag, max(masque, sansmasque) * 100)
		putText(hog, tag, (startA, startB - 20),

			FONT_HERSHEY_DUPLEX, 0.45, tint, 2)
		rectangle(hog, (startA, startB), (endA, endB), tint, 2)

	imshow("Camera0", hog)

	if waitKey(1) & 0xFF == ord("x"):
		break

destroyAllWindows()
cam.stop()
