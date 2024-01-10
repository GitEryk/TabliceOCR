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
    img = cv2.imread("Assets/wzornik.png")
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
        # ??? dla literki I
        if index == 8:
            x = x - 5
            w = w + 10
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (maxwidth, maxheight))
        _, roi = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY_INV)
        chars[index] = roi

    return chars


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
        #imshow(imgM)
        #plt.title("Kadrowanie")
        #plt.show()
        return imgM
    else:
        imshow(img)
        plt.title("Nieczytelna rejestracja")
        plt.show()
        return None

def getCharBox(img_gray):
    chars = []
    count = 0
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 
        maxValue=255, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=9, 
        C=13)
    #imshow(img_thresh)
    #plt.title("THRESH ")
    #plt.show()
    while True:
        con, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        charBox = [cv2.boundingRect(contour) for contour in con]
        for cutChar in charBox:
            x,y,w,h = cutChar
            if (8<w<15 and 32<h<37) or (3<w<8 and 32<h<37) or (13<w<25 and 32<h<37):
                #print(f"IF: x{x} y{y} w{w} h{h}") #KASUJ
                #time.sleep(0.5)
                chars.append((x,y,w,h))
                y = 0
                h = 40
                img_thresh[y:y+h,x-1:x+w+1] = 0
                #imshow(img_thresh)
                #plt.title("CUT")
                #plt.show()
                meanG = np.mean(img_thresh)
               
        if count < 1:
            img_flip = cv2.flip(img_thresh,0)
            #imshow(img_flip)
            #plt.title(f"FLIP {count}")
            #plt.show()
            img_thresh = cv2.bitwise_or(img_thresh, img_flip)
            #imshow(img_thresh)
            #plt.title(f"FLIP THRESH {count}")
            #plt.show()
            
        else:
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel = np.ones((count,count), np.uint8))
            img_thresh = cv2.dilate(img_thresh, kernel = np.ones((count,count), np.uint8))
            img_thresh[:2,:] = 0
            img_thresh[-2:,:] = 0
            '''
            imshow(img_thresh)
            plt.title(f"ClOSE {count}")
            plt.show()          
            '''
        count = count + 1
        
        print(F"meanG {meanG}") #KASUJ
        if  meanG < 10:
            break
    return chars

    
# MAIN
templeChars = getTemple()
group_result = []
rawChars = []

img = cv2.imread(f"Assets/{1}.jpg")
tablica = findTable(img) #return img
if tablica is not None:
    img = tablica
    tablica = cv2.cvtColor(tablica, cv2.COLOR_BGR2GRAY)
    rawChars = getCharBox(tablica) #return point list

for i, rawChar in enumerate(rawChars):
    x, y, w, h = rawChar
    charR = tablica[y:y + h, x:x + w]
    charR = cv2.resize(charR, (34, 23))
    _, charR =cv2.threshold(charR, 140, 255, cv2.THRESH_BINARY_INV)
    predict = []
    for (char_name, charT) in templeChars.items(): 
        charT = cv2.resize(charT, (34, 23))
        result = cv2.matchTemplate(charR, charT, cv2.TM_CCOEFF)
        (_, score, _, _) = cv2.minMaxLoc(result)
        predict.append(score)
    group_result.append(str(np.argmax(predict)))
    
print(group_result)
    

for count, position in enumerate(group_result):
    charT = templeChars[int(position)]
    charT = cv2.resize(charT, (34, 23))
    (x,y,w,h) = rawChars[count]
    charR = tablica[y:y + h, x:x + w]
    charR = cv2.resize(charR, (34, 23))
    
    plt.subplot(221)
    imshow(charR)
    plt.title("Wycinek")

    plt.subplot(222)
    imshow(charT)
    plt.title("wzornik")
    plt.show()


napis = []
for x in group_result:
    x = int(x)
    if x < 26:
        napis.append(chr(x + 65))
    else:
        napis.append(chr(x + 22))

print(napis)



