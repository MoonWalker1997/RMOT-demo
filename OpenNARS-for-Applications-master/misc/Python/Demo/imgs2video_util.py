import cv2

img_root = "C:\\Users\\TORY\\Downloads\\MOT17\\MOT17\\train\\MOT17-09-DPM\\img1\\"
fps = 30
size = (1920, 1080)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_writer = cv2.VideoWriter("C:\\Users\\TORY\\Downloads\\MOT17\\MOT17\\train\\MOT17-09-DPM\\video.mp4", fourcc, fps, size)

for i in range(1, 526):
    frame = cv2.imread(img_root + str(i).zfill(6) + ".jpg")
    video_writer.write(frame)

video_writer.release()