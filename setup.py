import numpy as np
import argparse
from PIL import Image
import pandas as pd


def roi_mean(index,radius,array):
    b,a = index
    nx,ny = array.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= radius*radius
    return np.mean(array[mask])


def process_stack(stack, len_stack, filename, roi_list):

    activations = np.zeros(shape=(len_stack, len(roi_list)))
    count = 0
    for i in range(len_stack):

        if i % 200 == 0:
            print(str(i) + " stacks processed...")
        stack.seek(i)
        image_as_array = np.array(stack).astype(np.uint8)

        for region in roi_list:
            roi_no = roi_list.index(region)
            mean_roi = roi_mean(region, 6, image_as_array)
            activations[count, roi_no] = mean_roi
        count += 1
    df = pd.DataFrame(activations)
    df.to_csv('input/' + filename+'.csv')


if __name__ == "__main__":

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", help="path and name of output image file")
    # ap.add_argument("-h", "--help", help="Display help message")

    ap.add_argument("-i", "--input", help="path to the tif file")
    ap.add_argument("-r","--roi", help="path to roi coordinates")
    args = vars(ap.parse_args())

    roi = args['roi']
    roi_file = pd.read_csv(roi, header=None, sep=';')
    roi_list = roi_list = [(x, y) for x, y in zip(roi_file[1], roi_file[0])]

    stack = Image.open(args["input"])
    len_stack = stack.n_frames

    output_name = args['output']

    process_stack(stack, len_stack, output_name, roi_list)
