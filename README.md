# People Counter with Deep Learning
## ( Using Object Detection(YOLO_v3) and Object Tracking )

	This people counter implementation uses object detection at fixed frame intervals and use object tracking the rest of the time. The whole thing is implemented in Pyton 3.

## Object Detection
	
	The object detection is done by using YOLO_v3 implemented with openCV2. Normally Yolo v3 is very fast when implemented through darknet, and gives approx. 30fps, but when implemented using opencv it goes down to a mere 3-5fps. Hence object tracking has been added to speed things up a bit.

## Object Tracking
	
	Object tracking here is done using dlib library which keeps track of the objects in the frame by calculating the distances of new estimated positions of the objects(estimated from previous frame) from the positoins in the previous frame and saving the same id for the one having minimum distances.

#### *Note: Though with both detection and tracking working together, it is still slow and inefficient and can be improved in lot of ways.*

## How to Run
	
	Since it si still under progress, it is not in executable form.
	In order to run the code, the following libraries need to be installed for python 3.
	(Can be done using pip)
	1. OpenCV2 (>= v3.4.0)
	2. numpy
	3. scipy
	4. dlib
	5. imutils

##### And then download yolov3 weights file to model folder. Link is given below:
##### https://pjreddie.com/media/files/yolov3.weights 

	When all libraries are installed run head_count.py
	If you want to use web cam : 
        $python3 head_count.py  

	If you want to give an input image : 
		$python3 head_count.py --image [full_path_of_image_file] 
	
	If you want to give an input video : 
		$python3 head_count.py --video [full_path_of_video_file] 
