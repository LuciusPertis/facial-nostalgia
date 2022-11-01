# facial-nostalgia
# name change 2022 10 13 1425 
# face_detect_video_collage
# 2022 04 26 2005
#
#
# keeping it simple random generation of scale adjusted frames.
# face/head orientation tracking and smooth collage for later


# --------------
# Map -
#
# frameMarker() ->  maintains a list of frames with detected faces,
#                   face types and no. of faces  
#
# faceInstall() ->  use openCV to detect a few faces and side faces and catalog
#                   the with name/labels
#
# genSeq() ->       helper to generate random sequence of the video collage
#                   a list of frames and or list of clip data for camtasia
#
# SeqMarker() ->    set specific orientation of face at a specific frame time
#
# //canonical alignment
#
# --------------
