import numpy as np
import cv2
from sklearn.datasets import fetch_mldata
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from sklearn.manifold import TSNE
from ggplot import *

train_file = 'train_file.txt'
test_file = 'test_file.txt'
feature_file = 'fc7_places_cnn.txt'
image_names_list = 'image_names_list.txt'
dataset_file = 'dataset_us.txt'

classes = ['action','adventure', 'animation', 'biography', 'comedy', 'crime', 'drama' , 'family', 'fantasy', 'history', 'horror', 'music', 'musical', 'mystery', 'romance', 'sci-fi', 'sport', 'thriller', 'war', 'western']

for i, j in enumerate(classes):
    print j, i

def data_loader():

    X_train = np.loadtxt(open('testing_autoencoded.txt'))
    y_train = np.loadtxt(open('testing_labels.txt'))

    for i, j in enumerate(classes):
        print j, int(np.sum(y_train[:,i]))



    print "size of x_train and y_train", X_train.shape, y_train.shape


    flag = -1                                                           
    x1, y1, x2, y2 = -1,-1,-1,-1                                        
                                                                        
    for i in range(X_train.shape[0]):                                   
        print "Adding stacking for multiple labels", i                  
        ids = np.where(y_train[i,:]==1)[0]                              
        for j in range(ids.shape[0]):                                   
            if flag == -1:                                              
                x1 = X_train[i,:]                                       
                y1 = np.asarray([ids[j]])                               
                flag = 1                                                
            else:                                                       
                x1 = np.vstack((x1, X_train[i,:]))                      
                y1 = np.vstack((y1, np.asarray([ids[j]]) ))             
    X_train = x1                                                        
    y_train = y1                                                        
                                                                        
    x1, y1 = None, None                                                 
    print "Training data shape after stacking", X_train.shape, y_train.shape

    return X_train, -1, y_train, -1

print "Import Successful"

#Please modify the data_loader function according to your own data.
'''
1. Here X - is a numpy 2-d matrix where each row is a feature vector for a particular data point.
2. y - is a label matrix (a column matrix corresponding to each data point in the data matrix X.
(Here the labels are 0,1,..... depending upon the number of labels.)
(The above data loader even works for multi-label vectors. Modify your label matrix accordingly.)
'''

X_train, X_test, y_train, y_test = data_loader()

X = X_train
y = y_train

print X.shape, y.shape

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]


df = pd.DataFrame(X, columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

X, y = None, None

print 'Size of the dataframe: {}'.format(df.shape)

rndperm = np.random.permutation(df.shape[0])

pca_50 = PCA(n_components=100)
pca_result_50 = pca_50.fit_transform(df[feat_cols].values)

print 'Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_))

n_sne = 5000

time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#tsne_pca_results = tsne.fit_transform(pca_result_50[rndperm[:n_sne]])
tsne_pca_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)


print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

df_tsne = None
df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_pca_results[:,0]
df_tsne['y-tsne'] = tsne_pca_results[:,1]

chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) + geom_point(size=70,alpha=0.8) + ggtitle("tSNE dimensions colored by Digit")
chart.save('your_tsne.png')
