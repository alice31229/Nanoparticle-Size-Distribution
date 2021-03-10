import pandas as pd
from skimage import feature, measure, draw
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import imutils
import math
import numpy as np
import cv2
import os


# define HOG Parameters
orientations = 13
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3



# define the sliding window:
def sliding_window(image, stepSize, windowSize):# image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):# this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x: x + windowSize[0]])

def File_imgs_overall(file,dir):
	# load df for diameter
    df = pd.read_excel(dir+'/'+file, engine='openpyxl')
    
    d_R = df['Diameter(nm)']
   
    ax = sns.distplot(d_R, hist = False, kde = True,
                kde_kws = {'linewidth': 1})

    d_avg.append(np.mean(d_R))
    d_std.append(np.std(d_R))
    
    
    plt.title('Nanoparticle size distribution')
    plt.xlabel('Particle diameter(nm)')
    plt.ylabel('Density')
    plt.xlim(5, 30)


def detect(file,dir):
    
    # read image in
	img1 = cv2.imread(dir+'/'+file, cv2.IMREAD_COLOR)
    
    # Convert to grayscale.
	img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

	# Test the trained classifier on an image below!
	scale = 0
	detections = []


	scale_percent = 40 # percent of original size 100 in scale bar   80
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)


	# resize image
	img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

	# defining the size of the sliding window (has to be, same as the size of the image in the training data)
	(winW, winH)= (32,32) #(64,64)
	windowSize=(winW,winH)
	downscale=1.15 
	stepSize = 2
	
	# upload the pretrained SVM 
	model = joblib.load('model_new2.npy')

	for resized in pyramid_gaussian(img, max_layer=-1, downscale=1.15): # loop over each layer of the image that you take! 5, -1
	    # loop over the sliding window for each layer of the pyramid
	    for (x,y,window) in sliding_window(resized, stepSize=2, windowSize=(winW,winH)): # stepsize=10, 3
	        # if the window does not meet our desired window size, ignore it!
	        # adjust the iteration scale
	        if window.shape[0] != winH or window.shape[1] != winW: # ensure the sliding window has met the minimum size requirement
	            continue
	        #window=color.rgb2gray(window)
	        hog_hist = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)  # extract HOG features from the window captured
	        fds = hog_hist.reshape(1,-1)
	        
	        pred = model.predict(fds) # use the SVM model to make a prediction on the HOG features extracted from the window
	        
	        if pred == 1: # 2.0 discussion
	            if model.decision_function(fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6, 1.0
	                print("Detection:: Location -> ({}, {})".format(x, y))
	                print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))
	                
	                detections.append((math.floor(x * (downscale**scale)), math.floor(y * (downscale**scale)), model.decision_function(fds),
	                                   math.ceil(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
	                                   math.ceil(windowSize[1]*(downscale**scale))))
	            
	    scale+=1
    
	clone = resized.copy()

	LU = []
	RD = []

	rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # do nms on the detected bounding boxes
	sc = [score[0] for (x, y, score, w, h) in detections]

	print("detection confidence score: ", sc)
	sc = np.array(sc)
	pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)

	for (xA, yA, xB, yB) in pick:
	    cv2.rectangle(img1, (xA, yA), (xB, yB), (0,255,0), 3)
	    LU.append((xA, yA))
	    RD.append((xB, yB))

	cv2.imwrite(os.path.join(r'./detection_result', 'ML23x0.4s2_{}'.format(file), img1)) 

		

	diameter_pixel_o = []

	img = cv2.imread(dir+'/'+file, 0)
	scale_percent = 40 # percent of original size 100 in scale bar   80
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	# resize image
	img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    
	for i in range(len(LU)):
		# center of crop img: x and y coordinate
		x_1 = LU[i][0]
		y_1 = LU[i][1]
		x_2 = RD[i][0]
		y_2 = RD[i][1]
		# crop img  
		crop_img = img[y_1:y_2, x_1:x_2]

		shape = crop_img.shape

		# RANSAC

        	#closing
		kernel = np.ones((11,11), np.uint8)
		img2 = cv2.morphologyEx(crop_img, cv2.MORPH_CLOSE, kernel)

        	#sharpening
		kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
		img2 = cv2.filter2D(img2, -1, kernel)

        	#additional blurring to reduce the offset toward brighter region
		img_blurred = cv2.GaussianBlur(img2.copy(), (8*2+1,8*2+1), 1)

		#equalizing
		equ = cv2.equalizeHist(img_blurred)
		mean_val = round(equ[:,:].mean()*(2/3))

        	#binary 
		ret, th = cv2.threshold(img_blurred, mean_val, 250, cv2.THRESH_BINARY) 

		cannyMax = ret 

		#detect edge
		canny = cv2.Canny(img_blurred, mean_val*0.66, mean_val*1.33)

		coords = np.column_stack(np.nonzero(canny))

		model, inliers = measure.ransac(coords, measure.CircleModel,
		                                min_samples=3, residual_threshold=1,
		                                max_trials=150)
		                                

		if int(model.params[0]) + model.params[2] <= shape[0] and int(model.params[0]) - model.params[2] > 0 and int(model.params[1]) + model.params[2] <= shape[0] \
		and int(model.params[1]) - model.params[2] > 0:

		    diameter_pixel_o.append(2*model.params[2]*2.5*scale_bar)
		    

	# remove outliers (some false positive detection results)
	z = np.abs(stats.zscore(diameter_pixel_o))

	diameter_pixel = np.array(diameter_pixel_o)[z < 3]
	diameter_nm.extend(diameter_pixel)

	df = pd.DataFrame(diameter_pixel, columns =['Diameter(nm)'])
	df.to_excel(os.path.join(r'./diameter(nm)', '{}.xlsx'.format(file)))

    
    	# plot distribution 
	hist = plt.hist(diameter_pixel, edgecolor='black', bins=10, color='green')

	plt.title('Nanoparticle size distribution')
	plt.xlabel('Particle diameter(nm)')
	plt.ylabel('Count')
	plt.xlim(0, 35)
	#plt.ylim(0, 60)
	avg = np.around(np.mean(diameter_pixel),1)
	std = np.around(np.std(diameter_pixel),1)
	count = len(diameter_pixel)
	textstr = '\n'.join((
	        r'$\bar{X}_D=%.1f$' % (avg, ),
	        r'$S_D=%.1f$' % (std, ),
	        r'$n=%.f$' % (count, )))

	plt.text(34, max(hist[0]), textstr,  fontsize=11,
	        verticalalignment='top', horizontalalignment='left')

	plt.grid(axis='y', alpha=0.75)
	plt.savefig(os.path.join(r'./distribution_result', "{}".format(file)))
	plt.clf() # erase the overlapping hist.


directory = r'./test_img'
scale_bar = float(input("scale bar (nm/px): "))
diameter_nm = []
for file in os.listdir(directory):
	if file.endswith(".jpg") or file.endswith(".png"):
		detect(file,directory)
		print(os.path.join(directory, file) + ' done')
		#print(os.getcwd())
		#exit()
	else:
		continue

df_all = pd.DataFrame(diameter_nm, columns =['Diameter(nm)'])
df_all.to_excel(os.path.join(r'./', 'Overall.xlsx'))

directory_2 = r'./diameter(nm)'
d_avg = []
d_std = []

for file in os.listdir(directory_2):
	
	if file.endswith(".xlsx"):
		File_imgs_overall(file,directory_2)
		print(os.path.join(directory_2, file) + ' done')

	else:
		continue

dff = pd.read_excel('Overall.xlsx', engine='openpyxl')
diameter_nm = dff['Diameter(nm)']

n = len([name for name in os.listdir(directory_2) if os.path.isfile(os.path.join(directory_2, name))]) 
main = sns.distplot(diameter_nm, hist = False, kde = True, label="distribution of all {} images".format(str(n)),
                    kde_kws = {'linewidth': 3, "color": "k"})


lg = plt.legend(labels=['distribution of all {} images'.format(str(n))], bbox_to_anchor=(1.05, 1.0), loc='upper left')

        
plt.savefig(os.path.join(r'./', "hist_overlap_addmain"), 
            bbox_extra_artists=(lg,), 
            bbox_inches='tight')

fig, ax = plt.subplots()

true = ax.axhline(y=np.mean(diameter_nm), xmin=0.1, xmax=0.9, ls='--') 

d = ax.scatter(range(len(d_avg)), d_avg, alpha = 0.25, s=30, c='green') # avg of each image
d_1 = ax.scatter(range(len(d_avg)), np.array(d_avg)-np.array(d_std), alpha = 0.25, s=30, c='red')
d_2 = ax.scatter(range(len(d_avg)), np.array(d_avg)+np.array(d_std), alpha = 0.25, s=30, c='red')

for i in range(len(d_avg)):

    ax.plot([i,i], [d_avg[i]-d_std[i], d_avg[i]+d_std[i]], ls='-', alpha = 0.25, c='yellowgreen')

lg2 = plt.legend([d, d_1, true], ['mean of each image', 'mean±standard deviation of each image', "mean of all images"], bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.xlabel("Each {} image".format(str(n)), fontsize=13) 
plt.ylabel("Mean±std diameter(nm)", fontsize=13) 
plt.grid(alpha=0.75) 
plt.savefig(os.path.join(r'./', "diameter_mean"), 
            bbox_extra_artists=(lg2,), 
            bbox_inches='tight')






