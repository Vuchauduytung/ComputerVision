import cv2
import sys
import os
import re
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import pickle
import pyautogui
import time


def main(): 
    # typing_method = 'original'
    typing_method = 'command line'
    
    print('\nWelcom to CreateData.py')
    path = os.path.dirname(os.path.abspath(__file__))
    sound_filename = 'Notification.mp3'
    sound_path = os.path.join(path, r'reference-files\notification-sound', sound_filename)
    mixer.init()
    mixer.music.load(sound_path)
    created_data_folder_path = os.path.join(path, 'data\positive') 
    cascade_filename = 'haarcascade_frontalface_alt.xml'
    cascade_path = os.path.join(path, 'reference-files\cascade-file\haarcascades', cascade_filename)
    face = cv2.CascadeClassifier(cascade_path)
    generic_data_name = "face"
    cap = cv2.VideoCapture(0)
    # font = cv2.FONT_HERSHEY_TRIPLEX
    count = get_count(created_data_folder_path)
    image_size = get_image_size(typing_method)
    capturing_method = get_capturing_method(typing_method)
    start = time.perf_counter()
    while(True): 
        if time.perf_counter() - start > 10: 
            mixer.music.play()
            time.sleep(2)
            break
        frame = capture_image(capturing_method, cap)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray,
                                    minNeighbors=5,
                                    scaleFactor=1.1,
                                    minSize=(25,25))   
        # Save captured face bounding box
        for (x,y,w,h) in faces: 
            count += 1
            face_detected = frame[y: y+h,x: x+w]
            face_detected_scaled = cv2.resize(face_detected,
                                                image_size)
            # Image directory 
            created_data_filename = generic_data_name + '(' +  str(count) + ')' + ".jpg"
            created_data_path = os.path.join(created_data_folder_path, created_data_filename)
            cv2.imwrite(created_data_path, face_detected_scaled)    
        # Show face's region bounded box    
        for (x,y,w,h) in faces: 
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )                
    stop_capturing_image(capturing_method, cap)
    cv2.destroyAllWindows()
    
def get_count(direct_path): 
    all_filenames = os.listdir(direct_path)
    count = 0
    for filename in all_filenames: 
        new_count = int(re.findall(r'\d+', filename)[0])
        if new_count > count: 
            count = new_count
    return count
        
def get_image_size(typing_method): 
    if typing_method == 'original': 
        height_cmdl = input('\tPlease type image height: ')
        width_cmdl = input('\tPlease type image width: ')
    elif typing_method == 'command line': 
        height_cmdl = sys.argv[2]
        width_cmdl = sys.argv[3]      
    if not height_cmdl.isnumeric(): 
        raise Exception('Sorry, there is a typo:  invalid height')
    if not width_cmdl.isnumeric(): 
        raise Exception('Sorry, there is a typo:  invalid width')
    height = int(height_cmdl)
    width = int(width_cmdl)
    return (height, width)

def get_capturing_method(typing_method): 
    if typing_method == 'original': 
        capturing_method = input('\tPlease type capturing method:  ')
    elif typing_method == 'command line': 
        capturing_method = sys.argv[1]
    return capturing_method

def capture_image(capturing_method, cap): 
    if 'cam' == capturing_method.lower(): 
        _, frame = cap.read()
    elif 'screen' == capturing_method.lower(): 
        RGB_image = np.array(pyautogui.screenshot())
        frame = cv2.cvtColor(RGB_image, cv2.COLOR_RGB2BGR)
    return frame

def stop_capturing_image(capturing_method, cap): 
    if 'cam' == capturing_method.lower(): 
        cap.release()


if __name__ == "__main__": 
    main()