import cv2 #opencv kütüphansi


cam = cv2.VideoCapture(0) #kemra açma

detector=cv2.CascadeClassifier('Classifiers/face.xml') #yuz tanıma algritmesi çağrıma
i=0
offset=50 # soldan, sagdan, ustten ve alttan 50 piksel
# verinin egitimi icin bosluk birakilmaktadir.
name=input('Kişinin id nosunu girin:') #her bir kişi için ID değeri al
while True:
    ret, im =cam.read()#kameradan resim okunmakta
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)#RGB renklerin gray değişkene dönüştürmek için
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(150, 150), flags=cv2.CASCADE_SCALE_IMAGE)
    '''CascadeClassifier::detectMultiScale>>>>>>>>>>>>>
    Giriş görüntüsündeki farklı boyuttaki nesneleri 
    algılar. Algılanan nesneler bir dikdörtgenler 
    listesi olarak döndürülür
    
    cv2.CascadeClassifier.detectMultiScale
    (image[, scaleFactor[, minNeighbors[, flags
    [, minSize[, maxSize]]]]])
    
    parmetreler:
    image : görüntü olarak verilen gray degeri
    
    scaleFactor – resmi %20 oranında büyütmek için 
    (kucultmek icinde olabilir)
    
    minNeighbors – Arama alani 5x5 bir matrislik(kernel)
    boyutlarda arama yapilmaktadir
    
    flags – Goruntuyu olceklendirmek icin kullanilir
    
    minSize – Goruntude aranacak olan nesnenin minimum 
    degeri 150x150 dizi olarak ayarlanmistir. Bunun nedeni
    kucuk olarak hatalı tespit edilen nesneler ihmal icin
    min degeri kullanilir. 
    
    maxSize – Goruntude aranacak olan nesnenin maksimum 
    degeri 150x150 dizi olarak ayarlanmistir. Bunun nedeni
    buyuk olarak hatalı tespit edilen nesneler ihmal icin
    max degeri kullanilir.
    
    Çıkışlar:
    #Vector of rectangles where each rectangle contains the detected object.
    '''
    for(x,y,w,h) in faces:
        i=i+1
        cv2.imwrite("dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        '''
        cv2.imwrite ile veri seti oluşuturlmaktadir. 
        dataset klasorunun icersine face-ile baslayan
        + name ile kullanicidan verilen id numarası 
        + nokta ile birlestir i le resmin numarası 
        (1,2, ..50) + ile dosya uzantisi jpg olarak 
        verilmistir. 
        gray[y-offset:y+h+offset,x-offset:x+w+offset]
        ifadesi ile goruntu alinmaktadir
        gray degiskeninde yukarida verilen offset degerleri
        cikartilip/ eklenerek yuz ve arkasinda bosluk 
        birakilmaktadir.
        '''
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)#resimin çerçevesinin belirlendi ve rengi de belirlendi
        cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])#resimler alırkan pencre adı ve boyutu
        '''
        yukarida ggray ile ayarlanip kaydedilen resimleri
        sadece ekranda renkli gostermek icin kullanilmaktadir.
        '''
        
        cv2.waitKey(100)
    if i>50:#51 tane resim aldıktan sonra kamera kapatılacaktır
        cam.release()
        cv2.destroyAllWindows()
        break
        
