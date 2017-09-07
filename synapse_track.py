from PIL import Image
import numpy as np
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
from savitzky_golay import *

# function to process images in various ways [gaussian,denoising,threshold,erode]
def image_processing(frame,kernel):
    gray = np.array(frame).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(gray, None, 70.0, 7, 21)
    r,thr = cv2.threshold(den,50,255,cv2.THRESH_BINARY)
    ero = cv2.erode(thr, kernel, iterations=1)
    return thr

# function to calculate the mean of a region of interest [roi]
def roi_mean(index,radius,array):
    b,a = index
    nx,ny = array.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= radius*radius
    return (np.mean(array[mask]))

# function to show a processed stack
def show_stack(pics, test_arr,rois,thr = 150):




    new_arr = np.zeros(shape=(1800, 91))
    # this part uses neighbor values and threshold to classify activations
    for i in range(90):
        for j in range(len(test_arr[:, i]) - 1):
            if (test_arr[:, i][j] > thr) and (test_arr[:, i][j+1] > thr) \
                    and (test_arr[:, i][j-1] > thr):
                new_arr[:, i][j] = 1
            else:
                new_arr[:, i][j] = 0

    # this part shows the images to the user
    for i in range(len(new_arr)-1):
        pics.seek(i)
        pic = np.array(pics).astype(np.uint8)
        for j in range(len(new_arr[i])-1):
            if new_arr[i][j] == 1:
                roi_cox = rois[j]
                cv2.circle(pic,roi_cox,13,(255,0,0),2)
                cv2.putText(pic, str(j+1), roi_cox, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                            255)

        cv2.imshow(str(i),pic)
        cv2.waitKey()
        if cv2.waitKey(33) == ord('a'):
            break


# function to read a stack and save data to a csv

def read_stack(stack,start,end,roi_list):
    kernel = np.ones((5, 5), np.uint8)
    activations = np.zeros(shape=((end-start),len(roi_list)))
    count = 0
    for i in range(start,end):
        print(i,count)
        stack.seek(i)
        prc = image_processing(stack,kernel)

        for roi in roi_list:
            roi_no = roi_list.index(roi)
            mean_roi = roi_mean(roi,6,prc)
            activations[count,roi_no] = mean_roi
            # if mean_roi > 2*std(activations[:,roi_no])
        count += 1


    means_ = activations[:,74]
    df = pd.DataFrame(activations)
    df.to_csv('activations_denoise_threshold.csv')
    pp.plot(means_)
    pp.show()
    #
    # cv2.imshow(str(i),org)
    # cv2.waitKey()

# initialization function

if __name__ == "__main__":

    stack = Image.open('img/DFDI13006_g.tif')
    df = pd.read_csv('data/roicoordinates.csv',
                     header=None, sep=';')
    activs = pd.read_csv('activations_denoise_threshold.csv',index_col=0)
    roi_list = [(x, y) for x, y in zip(df[1], df[0])]
    show_stack(stack,np.array(activs),roi_list)
    # read_stack(stack, 0, 1800,roi_list)






