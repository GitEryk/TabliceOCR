import cv2
import matplotlib.pyplot as plt
import numpy as np

def imshow(image):
  if len(image.shape) == 2 or len(image.shape) == 3 and image.shape[-1] == 1:
    plt.imshow(image, cmap="gray")
  else:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def compare_vertices(vertex):
    x, y = vertex
    return x + y

def readChar(img):
    repat = True
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imshow(img_gray)
    plt.title("GRAY")
    plt.show()
    while repat:
        ret, img_thresh = cv2.threshold(img_gray, 120, 255, type=cv2.THRESH_BINARY_INV)
        imshow(img_thresh)
        plt.title("THRESH")
        plt.show()
        con, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(con))
        charBox = [cv2.boundingRect(contour) for contour in con]
        
        for char in charBox:
            x,y,w,h = char
            if 5<w<15 and 15<h<35:
                #print(f"IF: x{x} y{y} w{w} h{h}")
                h = 30
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
                repat = False
            else:
                #print(f"ELSE x{x} y{y} w{w} h{h}")
                if w<10:
                    if y < 20:    
                        h = h+5
                    else:
                        y = y-5
                    imshow(cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2))
                    plt.show()
                    img_gray = cv2.rectangle(img_gray, (x,y), (x+w,y+h), (0,0,0), 1)
                    imshow(img_gray)
                    plt.title("img_gray img_gray")
                    plt.show()
                    repat = True
                
            imshow(img)
            plt.show()
        
    '''
    for i in range(0,120,10):
        imgCut = img_gray[:,i:i+10]
        imshow(imgCut)
        plt.title(f"CUT {i}-{i+10}")
        plt.show()
    '''
def OCR(img):
    img = cv2.resize(img, (120,40))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgL = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_gray = cv2.equalizeHist(img_gray)
    # WAŻNE WYŚWIETL !!!!!!
    #imshow(img)
    #plt.title("Oryginalne zdjęcie")
    #plt.show()
    
    flaga = False
    for thresh in range(190,140,-5):
        ret, img_thresh = cv2.threshold(img_gray, thresh, 255, type=cv2.THRESH_BINARY)
        #imshow(img_thresh)
        #plt.title("THRESH")
        #plt.show()
        for i in range(1,9):
            close_img = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel = np.ones((i,i), np.uint8))
            open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel = np.ones((i,i), np.uint8))
            '''
            plt.subplot(221)
            imshow(close_img)
            plt.title("CLOSE")
            plt.subplot(222)
            imshow(open_img)
            plt.title("OPEN")
            plt.show()
           '''
            con, _ = cv2.findContours(open_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            box = []
            for contour in con:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    approx2 = approx.reshape(4,2)
                    approx2 = sorted(approx2, key=compare_vertices)
                    w = np.linalg.norm(approx2[2] - approx2[0])
                    h = np.linalg.norm(approx2[1] - approx2[0])
                    #print(f"{i}. w {round(w,2)} h {round(h,2)}")
                    if 80<w<120 and 20<h<40:
                        cv2.drawContours(imgL, [approx], 0, (0, 0, 255), 1)
                        box.append(approx2)
                        flaga = True
                        break
            if flaga:
                break
        if flaga:
            break
    
    if flaga:
        # WAŻNE WYŚWIETL !!!!!!
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

# 1..13
for i in range(1,2):
    img = cv2.imread(f"Assets/{i}.jpg")
    tablica = OCR(img)

    if tablica is not None:
        readChar(tablica)














