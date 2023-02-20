# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 15:41:21 2023

@author: payam
"""

import cv2
import mediapipe as mp
import time





class faceMark_detector:
    
    def __init__(self,static_mode = False,max_faces = 10,refine_landmark = False,min_detection_conf = 0.5,min_tracking_conf = 0.5):
        
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.refine_landmark = refine_landmark
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf

        self.mpD = mp.solutions.drawing_utils
        
        self.mp_facemesh = mp.solutions.face_mesh
        
        self.mesh = self.mp_facemesh.FaceMesh(self.static_mode, self.max_faces,
                                              self.refine_landmark,
                                              self.min_detection_conf,
                                              self.min_tracking_conf)
        
        self.draw_spc = self.mpD.DrawingSpec(thickness = 1, circle_radius= 1)

    def find_marks(self,img,draw = True):
        
        self.img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        self.res = self.mesh.process(self.img_RGB)
        
        f_aces =[]
        if self.res.multi_face_landmarks :
            
            for facelms in self.res.multi_face_landmarks:
                if draw:
                    self.mpD.draw_landmarks(img,facelms,self.mp_facemesh.FACEMESH_CONTOURS,self.draw_spc,self.draw_spc)
                        
                
                f_ace = []
                for id,L in enumerate(facelms.landmark):
                    # print(L)
                    h,w,c = img.shape
                    x,y = int(L.x * w), int(L.y * h)
                    cv2.putText(img, str(id),(x,y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,255,0), 1)
                    
                    # print(id,x,y)
        
                    f_ace.append([id,x,y])
                f_aces.append(f_ace)

        return img,f_aces


def main():
    
    cam = cv2.VideoCapture("http://192.168.1.102:4747/video")

    ptime = 0
    
    detec = faceMark_detector()
    while True : 
        success,img = cam.read()
        
        img,faces = detec.find_marks(img)
        
        if len(faces)!=0:
            print(faces[0][1][1]-faces[0][9][1],faces[0][1][2]-faces[0][9][2])
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img,f' FPS :{int(fps)}',(20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3 )
        
        
        cv2.imshow("IMAGE",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break





if __name__ == "__main__":
    main()
    
cv2.destroyAllWindows()








