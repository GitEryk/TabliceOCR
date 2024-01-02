import cv2
import matplotlib.pyplot as plt
import numpy as np

def imshow(image):
  if len(image.shape) == 2 or len(image.shape) == 3 and image.shape[-1] == 1:
    plt.imshow(image, cmap="gray")
  else:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

img = cv2.imread("4.jpg")
plt.title("Oryginalne zdjęcie")

img = cv2.resize(img, (120,40))
mean = np.mean(img)
print(mean)
if mean < 105:
    img += 20
    print("+20 do zdjęcia")
imshow(img)
plt.show()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgL = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


#180
ret, img_thresh = cv2.threshold(img_gray, 170, 255, type=cv2.THRESH_BINARY)
imshow(img_thresh)
plt.show()


close_img = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel = np.ones((4,4), np.uint8))
imshow(close_img)
plt.show()


con = cv2.findContours(close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
con = con[0]
box = []
wsk = 0
for contour in con:
    (x,y,w,h) = cv2.boundingRect(contour)
    print(f"w{w} h{h}")
    ratio = w/h
    if 2.2<ratio<5.1:
        print(ratio)
        box.append((x,y,w,h))
        
pts1 = []

for (i, (x,y,w,h)) in enumerate(box):
    print(w)
    print(h)
    cv2.rectangle(imgL, (x,y), (x+w,y+h), color=(0,0,255))
    '''
    cv2.line(imgL, (x,y-2), (x+w,y+5), color=(0,0,255)) #góra
    cv2.line(imgL, (x+3,y+h-5), (x+w,y+h), color=(0,0,255)) #dół
    cv2.line(imgL, (x,y-2), (x+3,y+h-5), color=(0,0,255)) #lewa
    cv2.line(imgL, (x+w,y+5), (x+w,y+h), color=(0,0,255)) #prawa
    '''
    pts1 = [(x,y),(x+w,y),(x,y+h),(x+w,y+h)]

imshow(imgL)
plt.title("Zaznaczona rejestracja")
plt.show()

pts1 = np.float32(pts1)
pts2 = np.float32([[0,0],[80,0],[0,30],[80,30]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgM = cv2.warpPerspective(img, matrix, (80,30))

imshow(imgM)
plt.title("Kadrowanie")
plt.show()










