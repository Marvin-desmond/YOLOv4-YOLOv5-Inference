# https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo
import cv2
import numpy as np 
import time 

np.random.seed(42)
classes = open('coco.names').read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

def time_algorithm(func):               
    def wrapper(*args, **kwargs):            
        then = time.perf_counter()           
        outputs = func(*args, **kwargs)                
        now = time.perf_counter()            
        print(f"Time taken: {now - then}")   
        return outputs
    return wrapper                           

########################################################################
#     def say_whee():                        #     @time_algorithm     #
#         print("Whee!")                     #     def say_whee():     #
#     time_whee = time_algorithm(say_whee)   #         print("Whee!")  #
#     time_whee()                            #     say_whee()          #
########################################################################


cfg = "./pretrained_models/yolov4/yolov4.cfg"
weights = "./pretrained_models/yolov4/yolov4.weights"

@time_algorithm
def load_yolo(cfg=None, weights=None):
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    layer_names = net.getLayerNames()
    return layer_names[net.getUnconnectedOutLayers()[-1] - 1], net

ln, yolov4 = load_yolo(cfg=cfg, weights=weights)

def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))
    return image

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_to_blob = cv2.dnn.blobFromImage(image, 1/255.0, (512, 512), swapRB=True, crop=False)
    return image_to_blob

@time_algorithm
def yolo_inference(blob):
    yolov4.setInput(blob)
    outputs = yolov4.forward(ln)
    return outputs

@time_algorithm
def detections_on_image(outputs, image):
    h, w = image.shape[:2]
    for detection in outputs:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > 0.5:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            box = [x, y, int(width), int(height)]
            boxes.append(box)
            confidences.append(float(confidence))
            classIDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5 ,0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image 

def render_image(image):
    cv2.imshow('window', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

boxes = []
confidences = []
classIDs = []
path = "./inputs/horse.jpg"

image = read_image(path)

outputs = yolo_inference(
    preprocess_image(path)
)

render_image(
    detections_on_image(
    outputs, 
    image
    )
)
#%%
