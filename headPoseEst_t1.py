import cv2 as cv
import mediapipe as mp

import numpy as np

mp_face_mesh = mp.solutions.face_mesh
_face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

_camera_focal   = 1 * 1080
_camera_matrix  = np.array([[1080, 0, 720/2],
                            [0, 1080, 1080/2],
                            [0, 0, 1]] )

_camera_distort_matrix = np.zeros((4, 1), dtype=np.float64)

def setFaceMesh(fm_user):
    _face_mesh = fm_user
    
def getCVector(face_crop_img):
    """
        returns the canonical orientation of the face as vector
        non-facing faces should be low mangitude (to show poor confidence)
    """
    pass
    
#get face f
def getReducedCvector(image):
    #assume image writeble == false
    #assume color in RGB
    
    #compute
    results = _face_mesh.process(image)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    zf_3d = 3000
    if not results.multi_face_landmarks : return ()
    
    face_landmarks = results.multi_face_landmarks[0]
    
    for lm in face_landmarks.landmark:
        #scale landmarks to img
        #todo optimise with list compost
        x, y = int(lm.x * img_w), int(lm.y * img_h)

        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])

    
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
    x = angles[0]* 360
    y = angles[1]* 360
    z = angles[2]* 360
    
    return (x, y, z, face_3d[1], face_landmarks)



if __name__ == '__main__':
    import time
    
    mp_drawing = mp.solutions.drawing_utils

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        
        #compute
        
        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        start = time.time()
        CVector = getReducedCvector(image)
        if len(CVector) : x, y, z, nose_3d, face_landmarks = CVector
        else:
            cv.imshow('headPoseEst.py', image)
            cv.waitKey(5)
            continue
        end = time.time()
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        
        #start post processing
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
        p1 = (int( nose_3d[0] ), int( nose_3d[1] ))
        p2 = (int(nose_3d[0] + y*10), int(nose_3d[1] - x*10))

        cv.line(image, p1, p2, (255, 0, 0), 3)

        cv.putText(image, text, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        co = lambda r : str(np.round(r,2))
        cv.putText(image, 'x: ' + co(x), (500,  50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv.putText(image, 'y: ' + co(y), (500, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv.putText(image, 'z: ' + co(z), (500, 150), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        
        fps = 1/(end-start)
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

        

