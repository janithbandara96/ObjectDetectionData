import cv2
import os

cv2.namedWindow('Test House', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

# cascade files for part identification
part1_casc = cv2.CascadeClassifier('comp/cascade1.xml')
part2_casc = cv2.CascadeClassifier('comp/cascade2.xml')
part3_casc = cv2.CascadeClassifier('comp/cascade3.xml')
part4_casc = cv2.CascadeClassifier('comp/cascade4.xml')
part5_casc = cv2.CascadeClassifier('comp/cascade5.xml')
part6_casc = cv2.CascadeClassifier('comp/cascade6.xml')
part7_casc = cv2.CascadeClassifier('comp/cascade7.xml')
part8_casc = cv2.CascadeClassifier('comp/cascade8.xml')
part9_casc = cv2.CascadeClassifier('comp/cascade9.xml')
part10_casc = cv2.CascadeClassifier('comp/cascade10.xml')
part11_casc = cv2.CascadeClassifier('comp/cascade11.xml')

# cap = cv2.VideoCapture(0)
#img = cv2.imread(input_path)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# detect1 = part1_casc.detectMultiScale(gray, 1.3, 5)
# detect2 = part2_casc.detectMultiScale(gray, 1.3, 5)
# detect3 = part3_casc.detectMultiScale(gray, 1.3, 5)
# detect4 = part4_casc.detectMultiScale(gray, 1.3, 5)
# detect5 = part5_casc.detectMultiScale(gray, 1.3, 5)
# detect6 = part6_casc.detectMultiScale(gray, 1.3, 5)
# detect7 = part7_casc.detectMultiScale(gray, 1.3, 5)
# detect8 = part8_casc.detectMultiScale(gray, 1.3, 5)
# detect9 = part9_casc.detectMultiScale(gray, 1.3, 5)
# detect10 = part10_casc.detectMultiScale(gray, 1.3, 5)
# detect11 = part11_casc.detectMultiScale(gray, 1.3, 5)
#
#
# for (x,y,w,h) in detect1:
#     print('found1')
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
#     cv2.putText(img, 'Item 1', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#     cv2.imwrite(output_path, img)
# for (x,y,w,h) in detect2:
#     print('found2')
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
#     cv2.putText(img, 'Item 2', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#     cv2.imwrite(output_path, img)
# for (x,y,w,h) in detect3:
#     print('found3')
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
#     cv2.putText(img, 'Item 3', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#     cv2.imwrite(output_path, img)
# for (x,y,w,h) in detect4:
#     print('found4')
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
#     cv2.putText(img, 'Item 4', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#     cv2.imwrite(output_path, img)
# for (x,y,w,h) in detect5:
#     print('found5')
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
#     cv2.putText(img, 'Item 5', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#     cv2.imwrite(output_path, img)
# for (x,y,w,h) in detect6:
#     print('found6')
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
#     cv2.putText(img, 'Item 6', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#     cv2.imwrite(output_path, img)
# for (x,y,w,h) in detect7:
#     print('found7')
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
#     cv2.putText(img, 'Item 7', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#     cv2.imwrite(output_path, img)
# for (x,y,w,h) in detect8:
#     print('found8')
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
#     cv2.putText(img, 'Item 8', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#     cv2.imwrite(output_path, img)
# for (x,y,w,h) in detect9:
#     print('found9')
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
#     cv2.putText(img, 'Item 9', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#     cv2.imwrite(output_path, img)
# for (x,y,w,h) in detect10:
#     print('found10')
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
#     cv2.putText(img, 'Item 10', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#     cv2.imwrite(output_path, img)
# for (x,y,w,h) in detect11:
#     print('found11')
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
#     cv2.putText(img, 'Item 11', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#     cv2.imwrite(output_path, img)
#
#
# cv2.imshow("Test House", img)
#
# cv2.waitKey(0)

def detectPart(input_path, output_path):

    for filename in os.listdir(input_path):

        img = cv2.imread(input_path+'/'+filename)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detect1 = part1_casc.detectMultiScale(gray, 1.3, 5)
        detect2 = part2_casc.detectMultiScale(gray, 1.3, 5)
        detect3 = part3_casc.detectMultiScale(gray, 1.3, 5)
        detect4 = part4_casc.detectMultiScale(gray, 1.3, 5)
        detect5 = part5_casc.detectMultiScale(gray, 1.3, 5)
        detect6 = part6_casc.detectMultiScale(gray, 1.3, 5)
        detect7 = part7_casc.detectMultiScale(gray, 1.3, 5)
        detect8 = part8_casc.detectMultiScale(gray, 1.3, 5)
        detect9 = part9_casc.detectMultiScale(gray, 1.3, 5)
        detect10 = part10_casc.detectMultiScale(gray, 1.3, 5)
        detect11 = part11_casc.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detect1:
            print('found1')
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'Item 1', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (x, y, w, h) in detect2:
            print('found2')
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'Item 2', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (x, y, w, h) in detect3:
            print('found3')
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'Item 3', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (x, y, w, h) in detect4:
            print('found4')
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'Item 4', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (x, y, w, h) in detect5:
            print('found5')
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'Item 5', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (x, y, w, h) in detect6:
            print('found6')
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'Item 6', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (x, y, w, h) in detect7:
            print('found7')
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'Item 7', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (x, y, w, h) in detect8:
            print('found8')
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'Item 8', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (x, y, w, h) in detect9:
            print('found9')
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'Item 9', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        for (x, y, w, h) in detect10:
            print('found10')
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'Item 10', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

         #     for (x, y, w, h) in detect11:
         #     print('found11')
         #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
         #     cv2.putText(img, 'Item 11', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        cv2.imwrite(output_path + '/' + filename, img)
        cv2.imshow("Test House", img)
        cv2.waitKey(1)
input_path = 'input/item5'
output_path = 'output/output5'
detectPart(input_path, output_path)