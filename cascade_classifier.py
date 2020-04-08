import cv2

nose = cv2.CascadeClassifier("./cascade_models/haarcascade_mcs_nose.xml")
mouth = cv2.CascadeClassifier("./cascade_models/haarcascade_mcs_mouth.xml")
face = cv2.CascadeClassifier("./cascade_models/haarcascade_frontalface_alt.xml")

