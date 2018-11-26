import SelectiveSearch
import numpy as np
import cv2 as cv
#import classifier
from ClassifierModel import Predict
from FeatureCompute import generatePhi
from FeatureCompute import wordCnt

if __name__ == '__main__':
    filename = ''
    img = cv.imread(filename)
    regions = regionGenerate(img)
    data = np.float32([]).reshape(0,wordCnt)
    for items in regions:
        phi = generatePhi(items[1])
        np.append(data, phi.T, axis=0)
    # get response
    response = Predict(data)
    
    '''
    Stratage to pick regions
    '''

    '''
        use i to find location in regions
    '''
