import cv2

image = cv2.imread('images/5.jpg')

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

smiles = smile_cascade.detectMultiScale(image, scaleFactor=1.8,minNeighbors=20)

for(sx,sy,sw,sh) in smiles:
    cv2.rectangle(image,(sx,sy),((sx+sw),(sy+sh)), (0,255,0),4)

cv2.imshow("Smile",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
