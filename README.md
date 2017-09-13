# synapse_activations
Script to process TIFF stacks containing images of a mouse's brain cell, particularly the dendrite. This script aims to classify whether certain ROI's receive calcium as a result of an activation. 

The project consists of a setup.py, which converts the TIFF stack to an array containing region-of-interest averages during the stack, and analyze.py, which analyses and smoothes the aforementioned array and uses various techniques to classify activations. The algorithm takes into account for instance the acceleration, width and amplitude of the peak. 
