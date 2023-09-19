from f_utils import *
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import scipy.io
from sklearn.mixture import GaussianMixture

def load_dataset():    
    print("loading data.....")    
    data =np.load("dataset/two_classes_data.npy")
    x_axis_data = data[:,:151]
    y_axis_data = data[:,151:302]
    z_axis_data = data[:,302:453]
    
    
    #labels
    Labels =np.load("dataset/Binay_class_labels.npy")
    
#     #vlad
    def sliding_window(data):
        window_size = 40
        stride = 10
        return np.lib.stride_tricks.sliding_window_view(data,window_size)[::stride,:]
    
    
    def GMM(data):
        gmm = GaussianMixture(n_components=12, covariance_type='spherical').fit(data)
        return gmm.means_
    
    
    def find_NearestNeighbour(localdes,neighborhood):
        min_distance = float('inf')
        NearestNeighbour = neighborhood[0]
        NearestN_index = 0
        for i in range(len(neighborhood)):
            neighbor = neighborhood[i]
            distance = cv2.norm(localdes - neighbor)
            if distance < min_distance:
                min_distance = distance
                NearestNeighbour = neighbor
                NearestN_index = i 
                return NearestNeighbour, NearestN_index        
            
            
    def vlad (localdescriptors, centers):
        dimensions = len(localdescriptors[0])
        vlad_vector = np.zeros((len(centers), dimensions), dtype=np.float32)
        for localdescriptors in localdescriptors:
            nearest_center, center_index = find_NearestNeighbour(localdescriptors,centers)
            for i in range(dimensions):
                vlad_vector[center_index][i] += (localdescriptors[i] - nearest_center[i])
        # L2 Normalization            
        vlad_vector = cv2.normalize(vlad_vector,vlad_vector)
        vlad_vector = vlad_vector.flatten()
        return vlad_vector   
    
    
    xfeaturevector = np.zeros((11771,480))
    for i in range(x_axis_data.shape[0]):
        #sliding window in every row
        slide_vec = sliding_window(x_axis_data[i])
        #GMM in slide_vec shape 12X40
        codebook = GMM(slide_vec)
        #vlad implement retirn 480 dimensions
        v = vlad(slide_vec,codebook) 
        xfeaturevector[i] = v
        
        
    yfeaturevector = np.zeros((11771,480))
    for i in range(y_axis_data.shape[0]):
        #sliding window in every row
        slide_vec = sliding_window(y_axis_data[i])
        #GMM in slide_vec shape 12X40
        codebook = GMM(slide_vec)
        #vlad implement retirn 480 dimensions
        v = vlad(slide_vec,codebook) 
        yfeaturevector[i] = v
        
        
    zfeaturevector = np.zeros((11771,480))
    for i in range(z_axis_data.shape[0]):
        #sliding window in every row
        slide_vec = sliding_window(z_axis_data[i])
        #GMM in slide_vec shape 12X40
        codebook = GMM(slide_vec)
        #vlad implement retirn 480 dimensions
        v = vlad(slide_vec,codebook) 
        zfeaturevector[i] = v
        
        
    final_feature_vector = np.concatenate((xfeaturevector,yfeaturevector,zfeaturevector),axis=1)
              
    train_x, test_x, train_y, test_y = train_test_split(final_feature_vector, Labels, 
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=0)

    
    # print("train_x, train_y, test_x, test_y",train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
    train_x, val_x, train_t, val_t = train_test_split(train_x, train_y, test_size=0.2)

    train_x = normalize_data(train_x)
    val_x = normalize_data(val_x)
    test_x = normalize_data(test_x)   
    
    
    return train_x.T, train_t.T, val_x.T, val_t.T, test_x.T, test_y.T   
