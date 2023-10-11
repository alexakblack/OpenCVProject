# OpenCVProject
Open CV Project for AHCS - Put any face inside a logo

Chapters Used:
  1. Chapter 3 - Loading, Displaying, and Saving
     - cv2.imread()
  2. Chapter 5 - Drawing
     - cv2.rectangle()
  3. Chapter 6 - Image Transformations
     - cv2.resize()
     - image[] (cropping)
     - np.zeros() (numpy arrays)
     - bitwise_and()
     - masking with numpy arrays
     - cv2.cvtColor()
  4. Chapter 8 - Blurring
     - Gaussian blur was used in earlier versions of my project but not the final version
  5. Chapters 9, 10, and 11 and cv2.dnn()
     - Using cv2 dnn (deep nueral networks) there is built in edge detection and contouring
     - cv2.dnn.blobFromImage() uses thresholding, swapping of BGR values and resizing to create a 4-d version of the image
     - net.setInput(blob) sets the image as the new basis for the (deep neural) network (here using caffe)
     - detections = net.forward() makes it so all "detections" are being performed on the same "net" network
     - The detections then check for edges using the dnn caffe
     - After it detects, draws the contour box with cv2.rectangle()
