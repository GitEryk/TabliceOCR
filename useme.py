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

def OCR(img):
    img = cv2.resize(img, (120,40))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgL = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_gray = cv2.equalizeHist(img_gray)
    
    imshow(img)
    plt.title("Oryginalne zdjęcie")
    plt.show()
    
    
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
    
    imshow(imgL)
    plt.title("Zaznaczona rejestracja")
    plt.show()
    
    pts1 = np.float32(box[0])
    pts2 = np.float32([[0,0],[0,40],[120,0],[120,40]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgM = cv2.warpPerspective(img, matrix, (120,40))
    
    imshow(imgM)
    plt.title("Kadrowanie")
    plt.show()

