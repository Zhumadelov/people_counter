import numpy as np
import cv2
import datetime
bodyCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

cap = cv2.VideoCapture('video2.mp4')

textIn = 0
textOut = 0

ret, frame1 = cap.read()
ret, frame2 = cap.read()
count = 0
width = 800

def testIntersectionIn(x, y):
    res = -500 * x + 400 * y + 157500
    if ((res >= -550) and (res < 550)):
        print(str(res))
        return True
    return False


# def testIntersectionOut(x, y):
#     res = -500 * x + 400 * y + 180000
#     if ((res >= -550) and (res <= 550)):
#         print(str(res))
#         return True
#
#     return False




while True:
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(frame1, contours, -1, (0,255,0), 2)
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 12000:
            continue
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 2)
        #cv2.putText(frame1, "Status:{}".format("Movement"), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        print(x,y,w,h)
        cv2.line(frame1, (width-100, 450), (width+450, 450), (250, 0, 1), 2)  # blue line
        cv2.line(frame1, (width-100, 470), (width+450, 470), (0, 0, 255), 2)  # red line

        rectagleCenterPont = ((x + x + w) // 2, (y + y + h) // 2)
        cv2.circle(frame1, rectagleCenterPont, 1, (0, 0, 255), 5)


        if (testIntersectionIn(x // 2, y // 2)):
            textIn += 1

        # if (testIntersectionOut((x + w) // 2, (y + h) // 2)):
        #     textOut += 1

        cv2.putText(frame1, "In: {}".format(str(textIn)), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.putText(frame1, "Out: {}".format(str(textOut)), (10, 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame1, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame1.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow('video', frame1)



    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
