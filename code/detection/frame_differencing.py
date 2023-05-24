import os
import cv2
import numpy as np

# Charger le flux vidéo
cap = cv2.VideoCapture('media/video.mp4')

# Lire les deux premiers cadres
ret, frame1 = cap.read()
ret, frame2 = cap.read()

frame_num = 0
while cap.isOpened():
    frame_num += 1
    frame_file = os.path.join("result/frame_differencing", f'frame_{frame_num:04d}.jpg')

    # Calculer la différence absolue entre le cadre actuel et le cadre précédent
    diff = cv2.absdiff(frame1, frame2)

    # Convertir l'image en gris
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour améliorer la détection de contour
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Appliquer un seuil
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilater l'image seuillée pour combler les trous
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Trouver les contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours sur l'image originale
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Afficher l'image avec les contours
    # cv2.imshow("feed", frame1)
    # cv2.imwrite(frame_file, frame1)
    cv2.imwrite(frame_file, dilated)

    # Mettre à jour les images de référence
    frame1 = frame2
    ret, frame2 = cap.read()

    # Quitter si 'q' est appuyé
    if cv2.waitKey(40) == 27:
        break

    print(frame_num)

cv2.destroyAllWindows()
cap.release()
