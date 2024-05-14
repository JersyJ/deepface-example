# DeepFace


Deepface je framework pro rozpoznávání tváří a analýzu obličejových vlastností (věk, pohlaví, emocí a rasy) pro jazyk Python. 

Obsahuje modely: 

 * VGG-Face
 * FaceNet
 * OpenFace
 * DeepFace
 * DeepID
 * ArcFace
 * Dlib
 * SFace
 * GhostFaceNet.

## Instalace

Pro instalaci je potřeba mít nainstalovaný Python 3.8 nebo novější. Doporučuji pracovat v systému Linux vzhledem k limitaci a verze tensorflow package, ale je možné použít i Windows. 

Framework se vyskytuje pod názvem `deepface` na PyPI.

## Vlastnosti/Funkce


* Rozpoznávání tváří

Tato funkce umožňuje rozpoznat tváře na 2 obrázcích a verifikovat tak, pokud se jedná se stejné nebo jiné osoby.


* Analýza tváří

Tato funkce aplikuje několikrát rozpoznání tváří. Porovná obrázek se svojí databází a na základě toho vrátí list s výsledky.


* Embeddings
  
Tato funkce umožňuje získat vektorové reprezentace tváří z obrázků.


* Analýza atributů tváře

Tato funkce umožňuje získat informace o věku, pohlaví, rasě a emocích z obrázků tváří.


* Backendy detektoru tváří

DeepFace implementuje mnoho backendů pro detekci tváří.

    * Můžete si vybrat mezi:
        * OpenCV
        * Dlib
        * Ssd
        * MTCNN
        * fastmtcnn
        * RetinaFace
        * mediapipe
        * yolov8
        * yunet
        * centerface


* Analýza v reálném čase

Knihovna má v sobě zabudované funkce pomocí kterých můžete analyzovat tváře v reálném čase např. přes webkameru.


* API

Aby nemuselo všechno běžet na jednom stroji, je možné použít API, které běží na serveru a poskytuje výsledky zpracování obrázků vzdáleně. Je zde dostupný i Docker image, takže je možné provozovat několik instancí ve velkém clusteru např. přes Kubernetes.


* CLI

Knihovna obsahuje i CLI, které umožňuje spouštět různé funkce z příkazové řádky. 

