# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


#from PIL import Image, ImageOps

# import the necessary packages
# rom tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# define globals
INIT_LR = 1e-4
EPOCHS = 20

# detects face in the frame, if found then detects mask
# takes frame, face network and mask network


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and create a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # frame2 = frame.copy()
    # image = blob
    # img2 = Image.fromarray(frame)
    # img2.save("img.jpeg")

    # pass blob through face network and print shape of detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print("Detection: " + str(detections.shape))

    # initialize lists for faces, their locations and predictions
    faces = []
    locs = []
    preds = []

    # pattern_preds = []

    # loop over the face detections
    for i in range(0, detections.shape[2]):
        # find the confidence value of detections
        confidence = detections[0, 0, i, 2]

        # compute bounding box only if confidence > 50%
        if confidence > 0.5:
            # compute the x,y coords of bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # keep bounding box inside frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face
            face = frame[startY:endY, startX:endX]
            # face_for_pattern = face
            # convert face from BGR to RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))  # resize face to 224x224
            face = img_to_array(face)
            # face_for_pattern = face
            face = preprocess_input(face)

            # append face and its location to lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # make prediction if there's at least one face detected
    if len(faces) > 0:
        # make batch predictions on a batch of 20 images
        #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=20)

    # return face locations and prediction values
    return (locs, preds)

    # Disable scientific notation for clarity
    # np.set_printoptions(suppress=True)

    # # Load the model
    # pattern_model = load_model('pattern_model.h5')

    # opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # pattern_model.compile(loss="binary_crossentropy", optimizer=opt,
    #                       metrics=["accuracy"])

    # data = np.ndarray(shape=(1, 224, 224, 3), dtype="float32")

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    # size = (224, 224)
    # image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # # turn the image into a numpy array
    # image_array = np.asarray(img2)

    # # display the resized image
    # image.show()

    # # Normalize the image
    # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # # Load the image into the array
    # data[0] = normalized_image_array

    # # run the inference
    # prediction = pattern_model.predict(data)
    # print(prediction)

    # pattern_preds.append(prediction)


# load pre-trained face model from disk
prototxtPath = 'deploy.prototxt.txt'
weightsPath = 'res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our own face mask detector model from disk
maskNet = load_model("mask_detector.model")

# start video stream on source 0 camera
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over frames in video stream
while True:
    # grab frame from thread and resize to 720 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=720)

    # detect faces and face masks
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over detected face locations and their prediction values
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions as a zip
        (startX, startY, endX, endY) = box
        # get mask and withoutMask values from prediction list
        (mask, withoutMask) = pred

        # set label and color for bounding box
        # green for mask, red for no mask
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        #label = label + pattern

        # find max value from mask and no mask values
        # output percentage and format it to two precision values
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame window with our names as title
    cv2.imshow("Face Mask Detection by Manal, Sarmad & Faizan", frame)
    key = cv2.waitKey(1) & 0xFF

    # press 'q' to exit execution
    if key == ord("q"):
        break

# destroy windows and stop video stream when execution stopped
cv2.destroyAllWindows()
vs.stop()
