import numpy as np
import cv2

cap = cv2.VideoCapture("C:\\Users\\TORY\\OneDrive - Temple University\\AGI research\\RMOT Demo\\RMOT-demo\\OpenNARS-for-Applications-master\\misc\\Python\\YOLOX_outputs\\yolox_x_mix_det\\track_vis\\2023_06_22_15_17_40\\liverpool.mp4")

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if ret == False:
        continue
    fgmask = fgbg.apply(frame)
    cv2.imshow("frame", fgmask)

    k = cv2.waitKey(10) & 0xFF

cap.release()
cv2.destroyAllWindows()