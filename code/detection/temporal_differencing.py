import os
import cv2
import numpy as np

cap = cv2.VideoCapture("media/video.mp4")

_, frame1 = cap.read()
_, frame2 = cap.read()

frame_num = 0
while True:
    frame_num += 1
    frame_file = os.path.join("result/temporal_differencing", f'frame_{frame_num:04d}.jpg')

    # Calcul de la différence absolue entre les deux cadres consécutifs
    diff = cv2.absdiff(frame1, frame2)

    # Conversion en niveau de gris
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Applique le flou pour réduire le bruit
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Seuil pour identifier les régions de mouvement significatif
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilatation pour plus de robustesse
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Trouve les contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Dessine un rectangle autour des contours
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.imshow("feed", frame1)
    cv2.imwrite(frame_file, frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

    print(frame_num)

cv2.destroyAllWindows()
cap.release()
