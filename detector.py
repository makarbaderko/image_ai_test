#In this python3 file we use a pretrained model to predict on images

#Import ObjectDetection class from the ImageAI library.
from imageai.Detection import ObjectDetection

#Creating an instance of the  image-ai detector
detector = ObjectDetection()

#Set up constant file paths
model_path = "./models/yolo-tiny.h5"
input_path = "./input/test45.jpg"
output_path = "./output/newimage.jpg"

#Let the computer know, that we''l be using the YOLOv3 model
detector.setModelTypeAsTinyYOLOv3()

#Load model
detector.loadModel()

#Detect
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

#Get probabilities for each category
for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])

