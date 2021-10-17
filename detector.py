import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer.py')#resimlerden elde edilen verileri dosyası çağırma

faceCascade = cv2.CascadeClassifier("Classifiers/face.xml");
#yüz tanıma siniflandiricisini cagir
path = 'dataSet'

cam = cv2.VideoCapture(0)#kamera açma

font = cv2.FONT_HERSHEY_SIMPLEX
#opencv den resim üzerinde yazılacak yazı tipini belirlendi

while True:
    ret, im =cam.read()#resim okuturma
    
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #RGB renklerin gray değişkene dönüştürmek için
    
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
    #flags=cv2.CASCADE_SCALE_IMAGE)
    '''
    Asagidaki dongu yuz bulundugu zaman donguye girilmektedir
    '''
    for(x,y,w,h) in faces:
        
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        '''
        kamera acildiktan sonra alinan yuz goruntusu ile 
        egitim asamasinda elde edilen model ile karsilastirilarak
        yuzun hangi id numarasina sahip oldugu belirlenmektedir
               
        '''
        cv2.rectangle(im,(x+50,y+50),(x+w+50,y+h+50),(225,0,0),2)
        #resimin çerçevesinin belirlendi ve rengi de belirlendi
        
        if(nbr_predicted==0):
            #eğer elde edilen resim ID değer 1 ise resin jalil'a aittir
             a='Dr.bekir'
        
        elif(nbr_predicted==1):
            #eğer elde edilen resim ID değer 1 ise resin osamah'a aittir
             a='Osamah'

        
        else:
            a='UNKOWNUN'
        
        '''
        cv2.putTextiel goruntu üzerinde ilk olarak kisi adi (a) ile
        yazilmaktadir. Sonra ... isareti eklenmektedir. Akabinde
        kisinin id  bilgileri, yukarida verilen fontun tipi
        (x,y+h) ile yazinin baslayacagi konumu, font ile
        yukarida belilenen yazi tipini, renk bilgisi ve son olarak
        cizgivkalinligi verilmiştir.
        '''
        cv2.putText(im,a+"..."+str(nbr_predicted), (x,y+h),font,1,(0,0,0),3) #yazı yazdırma
        
        cv2.imshow('im',im)
        #resim göster
    k = cv2.waitKey(30) & 0xff
    if k == 27:#esc buttonu ektif etme
        break

cam.release()
cv2.destroyAllWindows()







