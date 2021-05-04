#!/usr/bin/python
import os
import time
# from vimba import *
import cv2
from pyModbusTCP.client import ModbusClient

part1_casc = cv2.CascadeClassifier('main_casc/cascade1.xml')
part2_casc = cv2.CascadeClassifier('main_casc/cascade2.xml')
part3_casc = cv2.CascadeClassifier('main_casc/cascade3.xml')
part4_casc = cv2.CascadeClassifier('main_casc/cascade4.xml')
part5_casc = cv2.CascadeClassifier('main_casc/cascade5.xml')
part6_casc = cv2.CascadeClassifier('main_casc/cascade6.xml')
part7_casc = cv2.CascadeClassifier('main_casc/cascade7.xml')
part8_casc = cv2.CascadeClassifier('main_casc/cascade8.xml')
part9_casc = cv2.CascadeClassifier('main_casc/cascade9.xml')
part10_casc = cv2.CascadeClassifier('main_casc/cascade10.xml')
part11_casc = cv2.CascadeClassifier('main_casc/cascade11.xml')
partpik_casc = cv2.CascadeClassifier('main_casc/cascadepik.xml')

part1_1_casc = cv2.CascadeClassifier('sub_casc/cascade1.1.xml')
part1_2_casc = cv2.CascadeClassifier('sub_casc/cascade1.2.xml')
part1_3_casc = cv2.CascadeClassifier('sub_casc/cascade1.3.xml')
part2_1_casc = cv2.CascadeClassifier('sub_casc/cascade2.1.xml')
part2_2_casc = cv2.CascadeClassifier('sub_casc/cascade2.2.xml')
part2_3_casc = cv2.CascadeClassifier('sub_casc/cascade2.3.xml')
part3_1_casc = cv2.CascadeClassifier('sub_casc/cascade1.1.xml')
part3_2_casc = cv2.CascadeClassifier('sub_casc/cascade1.2.xml')
part3_3_casc = cv2.CascadeClassifier('sub_casc/cascade1.3.xml')
part4_1_casc = cv2.CascadeClassifier('sub_casc/cascade4.1.xml')
part4_2_casc = cv2.CascadeClassifier('sub_casc/cascade4.2.xml')
part5_1_casc = cv2.CascadeClassifier('sub_casc/cascade5.1.xml')
part5_2_casc = cv2.CascadeClassifier('sub_casc/cascade5.2.xml')
part6_1_casc = cv2.CascadeClassifier('sub_casc/cascade6.1.xml')
part6_2_casc = cv2.CascadeClassifier('sub_casc/cascade6.2.xml')
part6_3_casc = cv2.CascadeClassifier('sub_casc/cascade6.3.xml')
part6_4_casc = cv2.CascadeClassifier('sub_casc/cascade6.4.xml')
part6_up_casc = cv2.CascadeClassifier('sub_casc/cascade6.up.xml')
part6_dowm_casc = cv2.CascadeClassifier('sub_casc/cascade6.down.xml')
part7_1_casc = cv2.CascadeClassifier('sub_casc/cascade7.1.xml')
part7_2_casc = cv2.CascadeClassifier('sub_casc/cascade7.2.xml')
part8_1_casc = cv2.CascadeClassifier('sub_casc/cascade8.1.xml')
part8_2_casc = cv2.CascadeClassifier('sub_casc/cascade8.2.xml')
part8_3_casc = cv2.CascadeClassifier('sub_casc/cascade8.3.xml')
part8_4_casc = cv2.CascadeClassifier('sub_casc/cascade8.4.xml')
part9_1_casc = cv2.CascadeClassifier('sub_casc/cascade9.1.xml')
part9_2_casc = cv2.CascadeClassifier('sub_casc/cascade9.2.xml')
part9_3_casc = cv2.CascadeClassifier('sub_casc/cascade9.3.xml')
part9_4_casc = cv2.CascadeClassifier('sub_casc/cascade9.4.xml')
part10_1_casc = cv2.CascadeClassifier('sub_casc/cascade10.1.xml')
part10_2_casc = cv2.CascadeClassifier('sub_casc/cascade10.2.xml')
part10_3_casc = cv2.CascadeClassifier('sub_casc/cascade10.3.xml')
part10_4_casc = cv2.CascadeClassifier('sub_casc/cascade10.4.xml')
part11_1_casc = cv2.CascadeClassifier('sub_casc/cascade11.1.xml')
part11_2_casc = cv2.CascadeClassifier('sub_casc/cascade11.2.xml')
part11_3_casc = cv2.CascadeClassifier('sub_casc/cascade11.3.xml')
part11_4_casc = cv2.CascadeClassifier('sub_casc/cascade11.4.xml')

cv2.namedWindow('Indust Labs Object Detection Program', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

def resize(img, scale):
    scale_percent = scale
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

def analyze1(cropped):
    orientation = 0
    detect1_1 = part1_1_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect1_2 = part1_2_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect1_3 = part1_3_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    if len(detect1_1)>0:
        orientation = 1
    elif len(detect1_2)>0:
        orientation = 2
    elif len(detect1_3)>0:
        orientation = 3
    detectpik = partpik_casc.detectMultiScale(cropped, 1.1, 1, minSize=(10, 10))
    if len(detectpik)>0:
        pick_point = detectpik[0]
        return orientation, pick_point

def analyze2(cropped):
    orientation = 0
    detect2_1 = part2_1_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect2_2 = part2_2_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect2_3 = part2_3_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    if len(detect2_1)>0:
        orientation = 1
    elif len(detect2_2)>0:
        orientation = 2
    elif len(detect2_3)>0:
        orientation = 3
    detectpik = partpik_casc.detectMultiScale(cropped, 1.1, 1, minSize=(10, 10))
    if len(detectpik)>0:
        pick_point = detectpik[0]
        return orientation, pick_point

def analyze3(cropped):
    orientation = 0
    detect3_1 = part3_1_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect3_2 = part3_2_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect3_3 = part3_3_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    if len(detect3_1)>0:
        orientation = 1
    elif len(detect3_2)>0:
        orientation = 2
    elif len(detect3_3)>0:
        orientation = 3
    detectpik = partpik_casc.detectMultiScale(cropped, 1.1, 1, minSize=(10, 10))
    if len(detectpik)>0:
        pick_point = detectpik[0]
        return orientation, pick_point

def analyze4(cropped):
    orientation = 0
    detect4_1 = part4_1_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect4_2 = part4_2_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    if len(detect4_1)>0:
        orientation = 1
    elif len(detect4_2)>0:
        orientation = 2
    return orientation

def analyze5(cropped):
    orientation = 0
    detect5_1 = part5_1_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect5_2 = part5_2_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    if len(detect5_1)>0:
        orientation = 1
    elif len(detect5_2)>0:
        orientation = 2
    return orientation

def analyze6(cropped):
    orientation = 0
    detect6_1 = part6_1_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect6_2 = part6_2_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect6_3 = part6_3_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect6_4 = part6_4_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect6_up = part6_up_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect6_down = part6_dowm_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))

    if len(detect6_1)>0:
        orientation = 1
    elif len(detect6_2)>0:
        orientation = 2
    elif len(detect6_3)>0:
        orientation = 3
    elif len(detect6_4)>0:
        orientation = 4
    elif len(detect6_up)>0:
        orientation = 5
    elif len(detect6_down)>0:
        orientation = 6
    return orientation

def analyze7(cropped):
    orientation = 0
    detect7_1 = part7_1_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect7_2 = part7_2_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    if len(detect7_1)>0:
        orientation = 1
    elif len(detect7_2)>0:
        orientation = 2
    return orientation

def analyze8(cropped):
    orientation = 0
    detect8_1 = part8_1_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect8_2 = part8_2_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect8_3 = part8_3_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect8_4 = part8_4_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))

    if len(detect8_1)>0:
        orientation = 1
    elif len(detect8_2)>0:
        orientation = 2
    elif len(detect8_3)>0:
        orientation = 3
    elif len(detect8_4)>0:
        orientation = 4
    return orientation

def analyze9(cropped):
    orientation = 0
    detect9_1 = part9_1_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect9_2 = part9_2_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect9_3 = part9_3_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect9_4 = part9_4_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))

    if len(detect9_1)>0:
        orientation = 1
    elif len(detect9_2)>0:
        orientation = 2
    elif len(detect9_3)>0:
        orientation = 3
    elif len(detect9_4)>0:
        orientation = 4
    return orientation

def analyze10(cropped):
    orientation = 0
    detect10_1 = part10_1_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect10_2 = part10_2_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect10_3 = part10_3_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect10_4 = part10_4_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))

    if len(detect10_1)>0:
        orientation = 1
    elif len(detect10_2)>0:
        orientation = 2
    elif len(detect10_3)>0:
        orientation = 3
    elif len(detect10_4)>0:
        orientation = 4
    return orientation

def analyze11(cropped):
    orientation = 0
    detect11_1 = part11_1_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect11_2 = part11_2_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect11_3 = part11_3_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))
    detect11_4 = part11_4_casc.detectMultiScale(cropped, 1.1, 1, minSize=(50, 50))

    if len(detect11_1)>0:
        orientation = 1
    elif len(detect11_2)>0:
        orientation = 2
    elif len(detect11_3)>0:
        orientation = 3
    elif len(detect11_4)>0:
        orientation = 4
    return orientation


def sendData(partNo, pickpoinX, pickpointY, orientationNo, count):
    client.write_multiple_registers(nextreg, [partNo, pickpoinX, pickpointY, orientationNo])
    nextreg += 10
    client.write_single_register(partNo, count)


def findObjects(input_path, output_path):
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    count8 = 0
    count9 = 0
    count10 = 0
    count11 = 0

    img = cv2.imread(input_path)

    # with Vimba.get_instance() as vimba:
    #     cams = vimba.get_all_cameras()
    #     with cams[0] as cam:
    #         while True:
    #             frame = cam.get_frame()
    #
    #             frame.convert_pixel_format(PixelFormat.Bgr8)
                # img = frame.as_opencv_image()

                # img = resize(img, 60)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect1 = part1_casc.detectMultiScale(gray, 1.1, 1, minSize=(250,250))
    detect2 = part2_casc.detectMultiScale(gray, 1.1, 4, minSize=(100,100))
    detect3 = part3_casc.detectMultiScale(gray, 1.1, 5,minSize=(90,90))
    detect4 = part4_casc.detectMultiScale(gray, 1.1, 5, minSize=(400, 400), maxSize=(900, 900))
    detect5 = part5_casc.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
    detect6 = part6_casc.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
    detect7 = part7_casc.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
    detect8 = part8_casc.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
    detect9 = part9_casc.detectMultiScale(gray, 1.1, 5, minSize=(500, 500))
    detect10 = part10_casc.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
    detect11 = part11_casc.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))

    for (x, y, w, h) in detect1:
        count1 += 1
        sizeText = "("+str(w)+","+str(h)+")"
        #"35-1205-03" 99-402019000
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cropped = img[y:y + h, x:x + w]
        orientation, pick_point = analyze1(cropped)
        pickpoint = (pick_point[0]+(pick_point[2]/2), pick_point[1]+(pick_point[3]/2))
        cv2.rectangle(img, (x+pick_point[0], y+pick_point[1]), ( x+pick_point[0]+pick_point[2], y+pick_point[1]+pick_point[3]), (0, 255, 0), 5)
        cv2.putText(img, "Part 1" + sizeText + "ori:"+ str(orientation) + str(pickpoint), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        # print("Part 1", end=",")
        sendData(1, pickpoint[0], pickpoint[1], orientation,count1)

    for (x, y, w, h) in detect2:
        count2 += 1
        sizeText = "("+str(w)+","+str(h)+")"
        #"35-1205-03"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cropped = img[y:y + h, x:x + w]
        orientation, pick_point = analyze2(cropped)
        pickpoint = (pick_point[0] + (pick_point[2] / 2), pick_point[1] + (pick_point[3] / 2))
        cv2.rectangle(img, (x+pick_point[0], y+pick_point[1]), ( x+pick_point[0]+pick_point[2], y+pick_point[1]+pick_point[3]), (0, 255, 0), 5)
        cv2.putText(img, "Part 2" + sizeText + "ori:"+ str(orientation) + str(pickpoint), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x+int(w/2),y+int(h/2)), 5, (255, 0, 0), -1)
        # print("Part 2", end=",")
        sendData(2, pickpoint[0], pickpoint[1], orientation, count2)

    for (x, y, w, h) in detect3:
        count3 += 1
        sizeText = "("+str(w)+","+str(h)+")"
        #"35-1205-03"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cropped = img[y:y + h, x:x + w]
        orientation, pick_point = analyze3(cropped)
        pickpoint = (pick_point[0] + (pick_point[2] / 2), pick_point[1] + (pick_point[3] / 2))
        cv2.rectangle(img, (x+pick_point[0], y+pick_point[1]), ( x+pick_point[0]+pick_point[2], y+pick_point[1]+pick_point[3]), (0, 255, 0), 5)
        cv2.putText(img, "Part 3" + sizeText + "ori:"+ str(orientation) + str(pickpoint), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        # print("Part 3", end=",")
        sendData(3, pickpoint[0], pickpoint[1], orientation, count3)

        cv2.circle(img, (x+int(w/2),y+int(h/2)), 5, (255, 0, 0), -1)

    for (x, y, w, h) in detect4:
        count4 += 1
        sizeText = "("+str(w)+","+str(h)+")"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cropped = img[y:y + h, x:x + w]
        orientation = analyze4(cropped)
        pickpoint = (x+(w/2), y+(h/2))
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        cv2.putText(img, "Part 4"+sizeText + "ori:"+ str(orientation) + str(pickpoint), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        # print("Part 4", end=",")
        sendData(4, pickpoint[0], pickpoint[1], orientation, count4)

    for (x, y, w, h) in detect5:
        count5 += 1
        sizeText = "("+str(w)+","+str(h)+")"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cropped = img[y:y + h, x:x + w]
        orientation = analyze5(cropped)
        pickpoint = (x+(w/2), y+(h/2))
        cv2.putText(img, "Part 5"+sizeText + "ori:"+ str(orientation) + str(pickpoint), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 5", end=",")
        sendData(5, pickpoint[0], pickpoint[1], orientation, count5)

    for (x, y, w, h) in detect6:
        count6 += 1
        sizeText = "("+str(w)+","+str(h)+")"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cropped = img[y:y + h, x:x + w]
        orientation = analyze6(cropped)
        pickpoint = (x+(w/2), y+(h/2))
        cv2.putText(img, "Part 6"+sizeText + "ori:"+ str(orientation) + str(pickpoint), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 6", end=",")
        sendData(6, pickpoint[0], pickpoint[1], orientation, count6)

    for (x, y, w, h) in detect7:
        count7 += 1
        sizeText = "("+str(w)+","+str(h)+")"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cropped = img[y:y + h, x:x + w]
        orientation = analyze7(cropped)
        pickpoint = (x+(w/2), y+(h/2))
        cv2.putText(img, "Part 7"+sizeText + "ori:"+ str(orientation) + str(pickpoint), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 7", end=",")
        sendData(7, pickpoint[0], pickpoint[1], orientation, count7)

    for (x, y, w, h) in detect8:
        count8 += 1
        sizeText = "("+str(w)+","+str(h)+")"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cropped = img[y:y + h, x:x + w]
        orientation = analyze8(cropped)
        pickpoint = (x+(w/2), y+(h/2))
        cv2.putText(img, "Part 8"+sizeText + "ori:"+ str(orientation) + str(pickpoint), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 8", end=",")
        sendData(8, pickpoint[0], pickpoint[1], orientation, count8)

    for (x, y, w, h) in detect9:
        count9 += 1
        sizeText = "("+str(w)+","+str(h)+")"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cropped = img[y:y + h, x:x + w]
        orientation = analyze9(cropped)
        pickpoint = (x+(w/2), y+(h/2))
        cv2.putText(img, "Part 9"+sizeText + "ori:"+ str(orientation) + str(pickpoint), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 9", end=",")
        sendData(9, pickpoint[0], pickpoint[1], orientation, count9)

    for (x, y, w, h) in detect10:
        count10 += 1
        sizeText = "("+str(w)+","+str(h)+")"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cropped = img[y:y + h, x:x + w]
        orientation = analyze10(cropped)
        pickpoint = (x+int(w/2), y+int(h/2))
        cv2.putText(img, "Part 10"+sizeText + "ori:"+ str(orientation) + str(pickpoint), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, pickpoint, 5, (255, 0, 0), -1)
        # print("Part 10", end=",")
        sendData(10, pickpoint[0], pickpoint[1], orientation, count10)

    for (x, y, w, h) in detect11:
        count11 += 1
        sizeText = "("+str(w)+","+str(h)+")"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cropped = img[y:y + h, x:x + w]
        orientation = analyze11(cropped)
        pickpoint = (x+(w/2), y+(h/2))
        cv2.putText(img, "Part 11"+sizeText + "ori:"+ str(orientation) + str(pickpoint), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 11", end=",")
        sendData(11,pickpoint[0],pickpoint[1],orientation, count11)


    print()
    cv2.putText(img, "count: "+str(count1), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Indust Labs Object Detection Program", img)
    cv2.imwrite(output_path, img)
    cv2.waitKey(10)

# if cv2.waitKey(1) & 0xFF == ord('\r'):
#     break

# Delete files older than 30 days
def deleteOldFiles(path):
	try:
		timenow =  time.time()
		deadline =  timenow-(2592000)

		for filename in os.listdir(path):
			output_file_path = path + filename
			t = os.path.getctime(output_file_path)
			if t<deadline:
				try:
					os.remove(output_file_path)
				except OSError as e:  ## if failed, report it back to the user ##
					print ("Error: %s - %s." % (e.filename, e.strerror))

	except Exception as e:
		print(e)

# Driver Code
if __name__ == '__main__':
    path = "C:/Users/industlabs/Desktop/TestStage3/input/"
    output = "C:/Users/industlabs/Desktop/TestStage3/output/"

    global nextreg
    nextreg = 20

    deleteOldFiles(output)

    try:
        client = ModbusClient(host="192.168.188.1", port=502)
        client.open()
    except:
        print("Unable to initialize the modbus client or open the modbus connection.")
        print("Press enter to exit.")
        exitInput = input()
        exit(1)

    i = 1

    for filename in os.listdir(path):
        while True:
            # fetch first register
            reg1 = client.read_holding_registers(0)
            # check data bank reset
            if reg1==0:
                break

        output_file = str(i) + ".jpg"
        input_path = path + filename
        output_path = output + output_file
        try:
            findObjects(input_path, output_path)
            client.write_single_register(i)
            ## Try to delete the file ##
            try:
                os.remove(input_path)
            except OSError as e:  ## if failed, report it back to the user ##
                print("Error: %s - %s." % (e.filename, e.strerror))
        except:
            print("error in finding objects", input_path)
        time.sleep(1)
        i += 1

    client.close()
    cv2.destroyAllWindows()
    exit(0)