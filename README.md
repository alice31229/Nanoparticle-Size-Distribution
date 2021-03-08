# Nanoparticle-Size-Distribution

This is a pipeline for obtaining the nanoparticle size distribution output with the nanoparticle samples transmission electron microscopy(TEM) image as input. Three main steps are demonstrated as follows: 
Firstly, machine learning based method is applied to detect nanoparticle in an image. The Histogram of Oriented Gradients (HOG) is used as features and Support Vector Machines (SVM) as the classifier to detect nanoparticle in given image. 
Then RANSAC algorithm is used to measure the size of each nanoparticle in the bounding boxes from the nanoparticle detection result.
Finally, after measured by RANSAC algorithm, we gather the size of each nanoparticle and plot the nanoparticle size distribution. 
