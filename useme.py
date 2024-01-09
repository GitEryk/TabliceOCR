import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

def imshow(image):
  if len(image.shape) == 2 or len(image.shape) == 3 and image.shape[-1] == 1:
    plt.imshow(image, cmap="gray")
  else:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def compare_vertices(vertex):
    x, y = vertex
    return x + y

def getTemple():
    img = cv2.imread("Assets/wzornik.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img,127,255, cv2.THRESH_BINARY_INV)
    con, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    charsbox = [cv2.boundingRect(contour) for contour in con]
    charsbox = sorted(charsbox, key=lambda x: x[0])
    maxwidth = max([char[2] for char in charsbox])
    maxheight = max([char[3] for char in charsbox])
    print(f"maxw {maxwidth} maxh {maxheight}") #KASUJ
    chars = {}
    for (index, box) in enumerate(charsbox):
        (x,y,w,h) = box
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (maxwidth, maxheight))
        _, roi = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY_INV)
        chars[index] = roi
    return chars

def getChar(img_gray):
    chars = []
    count = 0
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 
        maxValue=255, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=9, 
        C=13)
    imshow(img_thresh)
    plt.title("THRESH ")
    plt.show()
    while True:
        con, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        charBox = [cv2.boundingRect(contour) for contour in con]
        for cutChar in charBox:
            x,y,w,h = cutChar
            if (8<w<15 and 32<h<37) or (3<w<8 and 32<h<37) or (13<w<25 and 32<h<37):
                print(f"IF: x{x} y{y} w{w} h{h}") #KASUJ
                y = 0
                h = 40
                time.sleep(1)
                chars.append((x,y,w,h))
                img_thresh[y:y+h,x-1:x+w+1] = 0
                imshow(img_thresh)
                plt.title("CUT")
                plt.show()
                meanG = np.mean(img_thresh)
               
        if count < 1:
            img_flip = cv2.flip(img_thresh,0)
            imshow(img_flip)
            plt.title(f"FLIP {count}")
            plt.show()
            img_thresh = cv2.bitwise_or(img_thresh, img_flip)
            imshow(img_thresh)
            plt.title(f"FLIP THRESH {count}")
            plt.show()
        else:
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel = np.ones((count,count), np.uint8))
            img_thresh = cv2.dilate(img_thresh, kernel = np.ones((count,count), np.uint8))
            img_thresh[:2,:] = 0
            img_thresh[-2:,:] = 0
            imshow(img_thresh)
            plt.title(f"ClOSE {count}")
            plt.show()          
        count = count + 1
        
        time.sleep(2) #KASUJ
        print(F"meanG {meanG}") #KASUJ
        if  meanG < 10:
            break
    return chars
        
def readChar():
    pass
    

def findTable(img):
    img = cv2.resize(img, (120,40))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgL = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_gray = cv2.equalizeHist(img_gray)
    # WAŻNE WYŚWIETL !!!!!!
    #imshow(img)
    #plt.title("Oryginalne zdjęcie")
    #plt.show()
    
    flaga = False
    for thresh in range(190,110,-5):
        ret, img_thresh = cv2.threshold(img_gray, thresh, 255, type=cv2.THRESH_BINARY)
        #imshow(img_thresh)
        #plt.title("THRESH")
        #plt.show()
        for i in range(1,9):
            close_img = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel = np.ones((i+2,i), np.uint8))
            open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel = np.ones((i,i+1), np.uint8))
     
            '''
            plt.subplot(231)
            imshow(img_thresh)
            plt.title(f"img_thresh {thresh}")
            plt.subplot(232)
            imshow(close_img)
            plt.title(f"CLOSE {i}")
            plt.subplot(233)
            imshow(open_img)
            plt.title(f"OPEN {i}")
            plt.show()
           '''
          
            con, _ = cv2.findContours(open_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            box = []
            for contour in con:
                epsilon = 0.018 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    approx2 = approx.reshape(4,2)
                    approx2 = sorted(approx2, key=compare_vertices)
                    w = np.linalg.norm(approx2[2] - approx2[0])
                    h = np.linalg.norm(approx2[1] - approx2[0])
                    #print(f"{i}. w {round(w,2)} h {round(h,2)}")
                    if 85<w<120 and 20<h<40:
                        cv2.drawContours(imgL, [approx], 0, (0, 0, 255), 1)
                        box.append(approx2)
                        flaga = True
                        break
            if flaga:
                break
        if flaga:
            break
    
    if flaga:
        #WAŻNE WYŚWIETL !!!!!!
        #imshow(imgL)
        #plt.title("Zaznaczona rejestracja")
        #plt.show()
        pts1 = np.float32(box[0])
        pts2 = np.float32([[0,0],[0,40],[120,0],[120,40]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgM = cv2.warpPerspective(img, matrix, (120,40))
        # WAŻNE WYŚWIETL !!!!!!
        imshow(imgM)
        plt.title("Kadrowanie")
        plt.show()
        return imgM
    else:
        imshow(img)
        plt.title("Nieczytelna rejestracja")
        plt.show()
        return None

# MAIN
chars = getTemple()
for i in  range(1,2):
    img = cv2.imread(f"Assets/{i}.jpg")
    tablica = findTable(img)

    if tablica is not None:
        tablica = cv2.cvtColor(tablica, cv2.COLOR_BGR2GRAY)
        rawChar = getChar(tablica)
        imshow(tablica)
        plt.show()
  
    print(len(rawChar)) #KASUJ
    group_result = []
    test = np.zeros((40,120,3), dtype=np.uint8) #KASUJ
    for i, char in enumerate(rawChar):
        x,y,w,h = char
        roi = tablica[y:y+h, x:x+w, :]
        predict = []
        for (char, charRoi) in chars.items():
            result = cv2.matchTemplate(roi, charRoi, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            predict.append(score)
        group_result.append(str(np.argmax(predict)))
        org_img = cv2.putText(test,"".join(group_result), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0),2)
        
        test[y:y+h, x:x+w, :] = tablica[y:y+h, x:x+w, :] #KASUJ
        imshow(test) #KASUJ
        plt.show() #KASUJ
  










