import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import argparse


def acceleration(avg):

    # function to compute acceleration by taking differences between
    # x[n] and x[n-stride]
    accelerations = avg.copy()
    smooth = avg.copy()
    stride = 3

    for i in range(len(accelerations[0])):
        acc = pd.Series(accelerations[:,i])

        # smooth the list using a rolling window
        smoothed_roi = acc.rolling(window=29).std()
        acc = np.array([smoothed_roi[i+stride] - smoothed_roi[i] for i in range(len(smoothed_roi)-stride)])
        acc = np.append(acc, ([0]*stride))
        smooth[:, i] = smoothed_roi
        accelerations[:, i] = acc

    find_activations(accelerations,avg, smooth,threshold)


def find_activations(acc, avg, smooth, thr):

    df = []
    for i in range(len(acc[0])):

        roi_data = acc[:, i]
        # create a mask where acceleration higher than threshold
        mask = roi_data > thr
        cuts = np.flatnonzero(np.diff(mask))  # find indices where mask changes
        cuts = np.hstack([0, cuts + 1, -1])

        # iterate over index pairs
        for a, b in zip(cuts[:-1], cuts[1:]):
            cut = acc[a:b,i].tolist()
            cut = cut[:cut.index(min(cut))]
            peak = a
            if len(cut) > peak_length:
                for k in range(1,10):
                    if roi_data[peak-k] < 1:
                        start_peak = peak - k
                    else: start_peak = peak
                # define end of activation as median of set of local minima in activation
                try:
                    localMin = int(argrelextrema(smooth[start_peak:, i], np.less,order=10)[0][0])
                except: localMin = 20
                end_peak = a + localMin
                amplitude = np.max(avg[start_peak:end_peak, i])
                peak_width = end_peak -start_peak
                if amplitude > 45:

                    print(i+1, start_peak, end_peak, peak_width, amplitude)

                    df.append({'ROI': i+1, 'start': start_peak, 'end': end_peak, 'amplitude': amplitude})

    pd.DataFrame(df, columns=['ROI', 'start',
                              'end', 'amplitude']).to_csv('output/' + output_filename+ '_activations.csv', sep=',')

if __name__ == "__main__":

    # construct argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", help="path to the tif file")
    ap.add_argument("-o", "--output", help="path and name of output image file",default='synapse_activations')
    ap.add_argument("-r", "--roi", help="path to region of interest coordinates")
    ap.add_argument("-p", "--peak_length", help="set a threshold the minimum length of an activation",
                    default=4, type=int)
    ap.add_argument("-c", "--radius", help="radius in which to find local minima to compute end of activation",
                    type=int, default=8)
    ap.add_argument("-w", "--window", help="window to use in rolling standard deviation function",
                    type=int, default=29)
    ap.add_argument("-t", "--threshold", help="threshold for acceleration mask",
                    type=int, default=9)
    args = vars(ap.parse_args())

    # create variables from command line parse
    input_filename = args['input']
    output_filename = args['output']
    roi = args['roi']
    radius = args['radius']
    peak_length = args['peak_length']
    window = args['window']
    threshold = args['threshold']
    input_file = pd.read_csv(input_filename, index_col=0)

    acceleration(np.array(input_file))
