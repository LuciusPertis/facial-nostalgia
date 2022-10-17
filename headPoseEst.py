import cv2 as cv
import mediapipe as mp

import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv.VideoCapture(0)

while cap.isOpened():
    sucess, image = cap.read()

    start = time.time()

    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)

    #improves performance
    image.flags.writeable = False

    results = face_mesh.process(image)
    #DONE ---

    #start post processing
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    zf_3d = 3000
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in (33, 263, 1, 61, 291, 199):

                    #extra binding for canonical orientation display
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * zf_3d)

                    #save into numpy array
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            #for each landmark -->
                    
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            #camera
            focal = 1 * img_w
            cam_matrix = np.array([ [focal, 0, img_h/2],
                                    [0, focal, img_w/2],
                                    [0, 0, 1]] )

            distort_matrix = np.zeros((4, 1), dtype=np.float64)

            #solve PnP
            success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, distort_matrix)
            rmat, jac = cv.Rodrigues(rot_vec)

            #get angels
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)
            x = angles[0]* 360 - 8
            y = angles[1]* 360
            z = angles[2]* 360

            if y < -10:
                text = "Looking Left"  
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x >10:
                text = "Looking Up"
            else:
                text = "Looking Forward"


            #display nosedirection
            node_3dProj, jac= cv.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, distort_matrix)

            p1 = (int( nose_2d[0] ), int( nose_2d[1] ))
            p2 = (int(p1[0] + y*10), int(p1[1] - x*10))

            cv.line(image, p1, p2, (255, 0, 0), 3)

            cv.putText(image, text, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            co = lambda r : str(np.round(r,2))
            cv.putText(image, 'x: ' + co(x), (500,  50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(image, 'y: ' + co(y), (500, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(image, 'z: ' + co(z), (500, 150), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            
        end = time.time()
        tot_time = end - start
        fps = 1/tot_time
        #print("FPS: ", fps)

        cv.putText(image, f'FPS: {int(fps)}', (20, 450), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
            image =         image,
            landmark_list = face_landmarks,
            connections =   mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = drawing_spec,
            connection_drawing_spec = drawing_spec)

        cv.imshow('headPoseEst.py', image)
        
        
        key = cv.waitKey(5)
        if key == 27: #esc
            break
        if key == ord(' '):
            cv.waitKey(0)
            

cap.release()
cv.destroyAllWindows()    

    
