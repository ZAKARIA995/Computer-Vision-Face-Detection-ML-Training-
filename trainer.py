import cv2
import os
import numpy as np
from PIL import Image 

recognizer = cv2.face.LBPHFaceRecognizer_create()
'''
face.createLBPHFaceRecognizer yuz tanimada kullanilan opencv 
kutuphanesinde yer alan bir algoritmadir. Her bir kullanicidan 
alinan 50 adet goruntu  face.createLBPHFaceRecognizer algoritmasinda
ozellik cikarimi (yuze ait noktalar) tanimlanmistir
'''

cascadePath = "Classifiers/face.xml" 

faceCascade = cv2.CascadeClassifier(cascadePath);

path = 'dataSet' # alinacak goruntulerin konumu (dataset klasoru)

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]


#image_paths degiskeniene dataset klasorundeki tum dosyaların konumu
#adi ve uzantisi bilgileri alinmaktadir. 
#

    #resimlerin dizi verileri için
    images = [] 
    #resimlerin ID değeri
    labels = [] 
    for image_path in image_paths: 
        
         # alinan goruntulerin sayisi kadar donmesi icin
         
         # resim oku ve  grayscale'ye dönüştür
        image_pil = Image.open(image_path).convert('L')
         
         
         # resim oku ve   numpy dizi dönüştür 
        image = np.array(image_pil, 'uint8')
         
         # resimden ID değeri okumak
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
#
#         Face-Id(kişi no).(resim no).jpg dosyasindan 
#         sadece resim kişi noyu almak icin kullanilir. Ornegin 
#         face-0.1.jpg dosyasinda 0 kişinin id'si 1 ise kişinin 
#         ikinci resmi oldugunu gostermektedir.
#         replace("face-", "") ile face- silinmektedir. Geriye
#         0.1.jpg kalmaktadir. 
#         image_path)[1] ile 0.1 ifadesi kalir.
#         split(".")[0]  ile noktaya gore ayir ve 0 deger olan
#         0 degerini alir. 
#
        print (nbr)
         
         # resim içindeki yüz algılamak için o da eğer resim içinde sadece yüz değil
        faces = faceCascade.detectMultiScale(image)
         
         # resimlerden elde edilen veriler images diziye aktar ve resimlerin ID sı labels diziye aktar
        for (x, y, w, h) in faces:
            

#             Goruntuler ve ID degerleri ayri ayri eklenmektedir

            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
             
            cv2.imshow("Egitim setine yuzler eklenmektedir...", image[y: y + h, x: x + w])
            cv2.waitKey(10)
     #  images  labels listesi dondurulmektedir.
    return images, labels


images, labels = get_images_and_labels(path)
'''
test icin alinan goruntuler gosterilmektedir.
'''
cv2.imshow('test',images[0])
cv2.waitKey(1)

recognizer.train(images, np.array(labels))
'''
verilen resimlerin verileri ve resim Id ları ile Yüz Tanıma 
algoritmasina gore egirilmektedir
'''
recognizer.save('trainer/trainer.yml') 
# egitim dosyalari kaydedilmektedir.
# model dosyalarinin genellikle uzantisi yml olmaktadir.
cv2.destroyAllWindows()
