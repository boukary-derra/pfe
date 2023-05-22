
""" =================> Notes ==================  """

# Initialisation ->cv2.createBackgroundSubtractorMOG2()

#   Aplliquer

#   Opérations morphologiques
            # -> cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # ->  cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

            # -> contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # -> cv2.contourArea(contour) > 500:  # Ignorer les petits contours
            # -> (x, y, w, h) = cv2.boundingRect(contour)
            # -> cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

""" =================>  START PROGRAM <==================  """
import os
import cv2
import numpy as np

# Initialiser le soustracteur de fond
soustracteur = cv2.createBackgroundSubtractorMOG2()

# Ouvrir le flux vidéo
cap = cv2.VideoCapture("media/video.mp4")
frame_num = 0
while True:
    frame_num += 1
    frame_file = os.path.join("result/background_subtraction", f'frame_{frame_num:04d}.jpg')
    ret, frame = cap.read()
    if not ret:
        break

    # Appliquer le soustracteur de fond
    fgmask = soustracteur.apply(frame)

    # Appliquer des opérations morphologiques pour réduire le bruit
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    # Détecter les contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours sur l'image originale
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignorer les petits contours
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #cv2.imshow('Original', frame)
    #cv2.imshow('Foreground', fgmask)
    cv2.imwrite(frame_file, frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # 'ESC' pour quitter
        break
    print(frame_num)

cap.release()
cv2.destroyAllWindows()
