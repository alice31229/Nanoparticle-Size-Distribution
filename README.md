# Nanoparticle-Size-Distribution

This is a pipeline for obtaining the nanoparticle size distribution output with the nanoparticle samples transmission electron microscopy(TEM) image as input. Three main steps are demonstrated as follows: 
- Firstly, machine learning based method is applied to detect nanoparticle in an image. The Histogram of Oriented Gradients (HOG) is used as features and Support Vector Machines (SVM) as the classifier to detect nanoparticle in given image. 
- Then RANSAC algorithm is used to measure the size of each nanoparticle in the bounding boxes from the nanoparticle detection result.
- Finally, after measured by RANSAC algorithm, we gather the size of each nanoparticle and plot the nanoparticle size distribution. 

The programming language is python, the following libraries are required:

1. Scikit-learn (For implementing SVM)
2. Scikit-image (For HOG feature extraction; RANSAC algorithm)
3. OpenCV (For testing; image processing library)
4. Numpy (Matrix multiplication)
5. Imutils (For Non-maximum suppression)
6. Matplotlib (For distribution plot)

A training set should comprise of:

Positive images: these images should contain only the object you are trying to detect
Negative images: these images can contain anything except for the object you are detecting
