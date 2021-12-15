import pytesseract as pyt
pyt.pytesseract.tesseract_cmd=r'C:\Users\HP Pavilion\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
import cv2
import pandas as pd
import time


image = cv2.imread('image3.jpg')
cv2.imshow("image3",image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray_scale",gray_image)

canny_edge = cv2.Canny(gray_image, 130, 300)
cv2.imshow("Canny_Edge",canny_edge)

(cnts, _) = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]


contour_with_license_plate = None
license_plate = None
x = None
y = None
w = None
h = None

for contour in cnts:
        perimeter = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            contour_with_license_plate = approx
            x, y, w, h = cv2.boundingRect(contour)
            license_plate = gray_image[y:y + h, x:x + w]
            break

license_plate = cv2.bilateralFilter(license_plate, 17, 17, 17)
(thresh, license_plate) = cv2.threshold(license_plate, 100, 210, cv2.THRESH_BINARY)
image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 231), 2)
cv2.imshow("lo",license_plate)

text = pyt.image_to_string(license_plate, config='--psm 6')
cv2.imshow("License Plate Detection", image)
print("License Plate :", text)
df={"Entry_Time": [time.asctime(time.localtime(time.time()))],"Car_Number": [text]}
df1=pd.DataFrame(df,columns=['Entry_Time','  ','Car_Number'])
df1.to_csv("DATA1.csv")
cv2.waitKey(0)