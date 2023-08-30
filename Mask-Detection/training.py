import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn as sklearn
from sklearn import model_selection
import matplotlib.pyplot as graph
from matplotlib.pyplot import style, plot 
import os
from numpy import arange

taux_apprentissage = float(10**-4)  #initial learning 
Epoche = 100
batch_size = 32 #byte size
dataset= ["Mask", "No_Mask"]
donnees = []
tags = []

"""Dans cette fonction, nous importons (à l'aide de os.path) et lisons (à l'aide de .load_image) notre ensemble de données d'image qui comprend deux 
catégories: 'Mask' et 'No_Mask'. Nous procedons ensuite a leur redimensionnement en 256X256 et les convertir en forme "array" (à l'aide de .img_to_array)  """
def import_dataset_et_lire_images(dataset, donnees): 

 print("Étape 1: Récupération du l'images...")

 for group in dataset: 

     group_path = os.path.join(r"C:\Users\SAMUEL KIMANI\Desktop\Final proj\dataset", group)
     for img in os.listdir(group_path):

         img_path = os.path.join(group_path, img)
         resized_image = keras.preprocessing.image.load_img(img_path, target_size=(256, 256)) #resizing all images to default(256)
         resized_image = keras.preprocessing.image.img_to_array(resized_image)  #convert to array format
         resized_image = keras.applications.mobilenet_v2.preprocess_input(resized_image)
        
         donnees.append(resized_image) #

         tags.append(group)
         
""" classe publique Spam contenant six fonctions"""
class Spam: 
 global bin, model,prétraitement,TrainA,testA, TrainB, testB, Train #Déclaration de variables globales

 """Dans cette fonction effectue l'encodage des images et devise le base de données en 'Train 80%' et 'test 20%'  """
 def encodage_et_splitDataset_train_test(test_size):
  print("Étape 2: Divisant l’ensemble de données en Train et test...")

  global bin, TrainA, testA, TrainB, testB
  bin = sklearn.preprocessing.LabelBinarizer() #binarization is called
  (TrainA, testA, TrainB, testB) = model_selection.train_test_split(np.array(donnees, dtype="float32"), #bcoz hidden layers accept array data only (1st argument of Traintestsplit)
                                                                          np.array(tf.keras.utils.to_categorical(bin.fit_transform(tags))), # array of the converterted binary matrix 
                                                                          test_size=0.20,train_size=0.80, #20% of our dataset is being used for testing
                                                                          stratify=np.array(tf.keras.utils.to_categorical(bin.fit_transform(tags))),#ensures both sets(Train,test) have atleast equal examples in each class in our array 
                                                                          random_state=42) # 42 recommended when using skicit - learn
  return (TrainA, testA, TrainB, testB)
 
 """ Dans cette fonction on effectue prétraitement de données en utilisans le parametre  'tf.keras.preprocessing.image.ImageDataGenerator' """
 def fct_prétraitement(rotation_range):
  print("Étape 3: Prétraitement en cours...")

  global prétraitement
  prétraitement = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20, #randomly rotate image anticlockwise
                                                            zoom_range=0.15,# float value for +15% zooming range of imageData
                                                            width_shift_range=0.2,
                                                            height_shift_range=0.2,
                                                            shear_range=0.15,
                                                            horizontal_flip=True,
                                                            fill_mode="nearest")
  return prétraitement

 """ Dans cette fonction, nous créons notre modèle CNN, chargeons MobileNetV2 classifieur dans le modèle de base avec des poids ImageNet pré-entraînés, 
 en veillant à ce que la couche FC de la tête soit laissée de côté."""
 #load MobileNetV2 with pre-Trained ImageNet weights , ensuring the head FC layer are left off   
 def création_model_avec_CNN(root):
  print("Étape 4: Construction du model CNN...")

  root = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, 
      input_tensor=tf.keras.layers.Input(shape=(256, 256, 3))) #three augments for RGB
      
  """ Construct le modèle de tête du modèle qui sera placé au-dessus du modèle de base: 'relu' est la couche d’activation et 'softmax' pour classification d'images"""
  tModel = root.output 
  tModel = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation="relu", input_shape=(256,256,3))(tModel)
  tModel = tf.keras.layers.MaxPooling2D(pool_size=(7, 7))(tModel)
  tModel = tf.keras.layers.Flatten(name="flatten")(tModel)
  tModel = tf.keras.layers.Dense(128, activation="relu")(tModel) 
  tModel = tf.keras.layers.Dropout(0.5)(tModel)
  tModel = tf.keras.layers.Dense(2, activation="softmax")(tModel)
  global model
  model = tf.keras.models.Model( inputs=root.input, outputs=tModel) 


  """ boucler et figer toutes les couches du modèle de base afin qu’elles ne soient pas mises à jour lors du premier 'Training' """
  for couche in root.layers: 
      couche.Trainable = False 
  return model

 """Cette fonction compile notre modèle CNN avec l'optimiseur Adam prêt pour Training.
  La décadence est calculée en divisant le taux d’apprentissage initial par le nombre d’epoches utilisés pendant le Training"""
  #now we compile our model with Adam optimizer 
 def compilation(optimizer):
  print("Étape 5: Complilation du model avec Adam optimizer...")

  optimizer = tf.keras.optimizers.Adam(learning_rate=taux_apprentissage, decay=taux_apprentissage / Epoche)
  model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]) #call optimizer, adam/rmsprop. metrics bcoz we have a balanced dataset
  model.summary()
  
 """ Dans cette fonction, nous apprendons la fonction construite avec 100 epoches à l’aide de la fonction model.fit()"""
 #Training phase
 print("TrainING of head net in progress")
 def phase_training(epochs):
  print("Étape 6:  Apprentissage de modèle construit avec 100 époques...")
  #model = tf.keras.models.Model( inputs=s.model_creation_with_CNN.root.input, outputs= s.model_creation_with_CNN.tModel)
  global Train
  Train = model.fit(
                   prétraitement.flow(TrainA,TrainB, batch_size = batch_size),
                   steps_per_epoch=len(TrainB) // batch_size,
                   validation_data=(testA,testB),
                   validation_steps=len(testA) // batch_size, epochs=Epoche
                  )

 """ Dans cette fonction, nous récupérons les index d’étiquettes des images d’index de la phase de test avec la probabilité prédite correspondante 
 et afficher le rapoer du classification"""
 #Make predictions on testing set
 def phase_test(predictionIndexes): 
  print("Étape 7: Phase du test et faites des prédictions...")
  predictionIndexes = model.predict(testA, batch_size = batch_size)

 #get index images' label indexes from the testing phase with corresponding predicted probability
  predictionIndexes = np.argmax(predictionIndexes, axis=1)
  
  print("Étape 8: Afficher le raport du classification...")
  print(sklearn.metrics.classification_report(testB.argmax(axis=1), predictionIndexes,target_names=bin.classes_))
  
  #Matrix de confusion
  matXconf = sklearn.metrics.confusion_matrix(testB.argmax(axis=1), predictionIndexes)
  count = sum(sum(matXconf))
  valid = (matXconf[0, 0] + matXconf[1, 1]) / count
  hypersens = matXconf[0, 0] / (matXconf[0, 0] + matXconf[0, 1])
  homogeneity = matXconf[1, 1] / (matXconf[1, 0] + matXconf[1, 1])

  print(matXconf)
  print("accuracy: {:.4f}".format(valid))
  print("savoir_faire: {:.4f}".format(hypersens))
  print("homogeneity: {:.4f}".format(homogeneity))

  matX= sklearn.metrics.ConfusionMatrixDisplay(matXconf)
  matX.plot()
 
#ploting the Training, loss and accuracy: metrics logged during Training
epoc = Epoche
def plot_Training(epoc, Train):
 print("Étape 9: Tracer le graphique de la formation et de la validation loss_accuracy...")
 
 style.use("ggplot")
 graph.figure()
 plot(arange(0, epoc), Train.history["loss"], label="Train_loss")
 plot(arange(0, epoc), Train.history["val_loss"], label="val_loss")
 plot(arange(0, epoc), Train.history["accuracy"], label="Train_acc")
 plot(arange(0, epoc), Train.history["val_accuracy"], label="val_acc")
 graph.title("Training & Validation Loss_Accuracy Grapgh")
 graph.xlabel("Epoches_Rounds #")
 graph.ylabel("Loss & Accuracy")
 graph.legend(loc="lower left")
 graph.savefig("Graphbrk100.png") #save graph as image in png format


#Appel du fonctions
import_dataset_et_lire_images(dataset, donnees)
s =Spam() 
s.encodage_et_splitDataset_train_test()
s.fct_prétraitement()
s.création_model_avec_CNN()
s.compilation()
s.phase_training()  
s.phase_test()
plot_Training(epoc,Train)

#Serialize the model to disk
print("Étape 10: Enregistrement du model...")
model.save("maskedfacebrk_100epoche.model", save_format ="h5")

