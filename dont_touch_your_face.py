import cv2
from detectors import *


def con_detect():
    path = "E:\Codes\PrathamAI\AI\Sample_Videos\My Test Video.mp4"
    #path = 0
    cap = cv2.VideoCapture(path)
    # https://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
    #cap.set(3, 1280/4)
    #cap.set(4, 1024/4)

    HandsDetector = TSDetector()
    FaceDetector = CVLibDetector()
    total_frames = 0
    handTouched = 0
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE) 
        frame = ResizeWithAspectRatio(frame, width=600, height=800) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        total_frames = total_frames + 1
        hands = HandsDetector.detect(rgb)
        face = FaceDetector.detect(rgb)
        
        
        if objects_touch(face, hands):
            img_detected = add_objects_to_image(rgb, hands)
            img_detected = add_objects_to_image(img_detected, face)
            handTouched = handTouched + 1
            
        if objects_touch(face, hands) == False:
            img_detected = add_objects_to_image(rgb, hands,color=(0, 0, 129))
            img_detected = add_objects_to_image(img_detected, face, color=(0, 255, 0))
        cv2.imshow('frame', cv2.cvtColor(img_detected, cv2.COLOR_RGB2BGR))
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    percentTouch = handTouched*100/total_frames
    print("Percentage Hand Touched: ",percentTouch)
    cap.release()
    cv2.destroyAllWindows()
   

if __name__ == '__main__':
    con_detect()
