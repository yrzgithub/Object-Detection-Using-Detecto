from detecto import core, utils, visualize

image = utils.read_image('camera.jpg')
model = core.Model()

labels, boxes, scores = model.predict_top(image)
visualize.show_labeled_image(image, boxes, labels)
print(labels)

# for project
import keyboard, cv2
from detecto import core, utils, visualize

model = core.Model()
camera = cv2.VideoCapture(0)
while not keyboard.is_pressed("esc"):
    ret, img = camera.read()
    cv2.imshow("project", img)
    cv2.waitKey(1)
    labels, boxes, scores = model.predict_top(img)
    print(labels)
    prob = str(scores).replace("tensor", "").strip("([").strip(")]").strip(" ")
    print(prob)
    prob = list(map(float, prob.split(",")))
    dictobjects = dict(zip(prob, labels))
    for i in dictobjects:
        if i > 0.9:
            print(dictobjects[i])
    print(dictobjects)
camera.release()
cv2.destroyAllWindows()

# project

from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows
from keyboard import is_pressed
from detecto import core
from pyttsx3 import init

convertor = init()

model = core.Model()
camera = VideoCapture(0)
while not is_pressed("esc"):
    ret, img = camera.read()
    imshow("project", img)
    waitKey(1)
camera.release()
destroyAllWindows()
labels, boxes, scores = model.predict_top(img)
prob = str(scores).replace("tensor", "").strip("([").strip(")]").strip(" ")
prob = list(map(float, prob.split(",")))
dictobjects = dict(zip(prob, labels))
for i in dictobjects:
    if i > 0.8:
        obj = dictobjects[i]
        print(obj)
        convertor.say(obj)
        convertor.runAndWait()
