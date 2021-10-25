"""
=============================================================================================
1.4 K-means Clustering using Human Activity Recognition Data (HAR)
url : http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
=============================================================================================
Dataset includes following activites :
1 WALKING
2 WALKING_UPSTAIRS
3 WALKING_DOWNSTAIRS
4 SITTING
5 STANDING
6 LAYING
=================================================

Performance measure of clustering:
            ARI (Adjusted Rand Score)
            NMI (Normalized Mutual Information)
==================================================
"""
#print(__doc__)

import numpy as np
import os
import urllib.request
import zipfile
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import KernelPCA
from scipy.spatial import Voronoi, voronoi_plot_2d

#*******************************************************
# prepare data, download from source and extract it 
#*******************************************************
def prepare_data():
    print("")
    
    data_file = "data/UCI HAR Dataset.zip"
    if os.path.exists(data_file):
        print("UCI HAR Dataset.zip already downloaded\n")
        
    else:
        print("Downloading UCI HAR Dataset.....")
        if not os.path.exists("data"):
            os.mkdir("data")
    
        data_source = "http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        urllib.request.urlretrieve(data_source, data_file)
        print("UCI HAR Dataset.zip downloaded")
        
        print("Extracting UCI HAR Dataset.zip....")
        with zipfile.ZipFile(data_file, "r") as zip_ref:
            zip_ref.extractall("data")
            print("UCI HAR Dataset prepared")
        print("")


def main():
    # get dataset 
    prepare_data()

    print("=====================================================")
    print("K-means Clustering ,Cluster Activities into 6 groups.")
    print("=====================================================\n")
    
    #load data
    X_train = np.loadtxt('/home/hpc/userName/Desktop/csc380_ai/HW6/data/UCI HAR Dataset/train/X_train.txt')
    y_train = np.loadtxt('/home/hpc/userName/Desktop/csc380_ai/HW6/data/UCI HAR Dataset/train/y_train.txt')
    X_test = np.loadtxt('/home/hpc/userName/Desktop/csc380_ai/HW6/data/UCI HAR Dataset/test/X_test.txt')
    y_test = np.loadtxt('/home/hpc/userName/Desktop/csc380_ai/HW6/data/UCI HAR Dataset/test/y_test.txt')
    
    # perform clustering using sklearn library KMeans
    kmeans_model = KMeans(6).fit(X_train)
    pred = kmeans_model.predict(X_test)
    
    # Calculate ARI measure
    print("===== ARI Measure ========")
    print(f'ARI: {adjusted_rand_score(y_test, pred):.3f}\n')
    
    # Calculate NMI measure
    print("====== NMI Measure =======")
    print(f'NMI: {normalized_mutual_info_score(y_test, pred):.3f}\n')
    
    #Kernel Principal component Analysis (KPCA)
    dim_reducer = KernelPCA(2, kernel='poly', degree=4)
    X_test_transformed = dim_reducer.fit_transform(X_test)
    
    #axis1 --> plot for true cluster , axis2 --> for KMeans results 
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18, 8))
    fig.suptitle('K-means Clustering HAR dataset (PCA-compressed)', fontsize=25)
    
    
    scatter_plot = ax1.scatter(*X_test_transformed.T, c=y_test, s=2)
   
    #lables from activity_info.txt 
    ax1.legend(scatter_plot.legend_elements()[0],
               ['WALKING','WALKING UPSTAIRS','WALKING_DOWNSTAIRS', 'SITTING', 'STANDING','LAYING'],
               loc='upper right')
    ax1.set_xlabel('Principal Component 1'); ax1.set_ylabel("Principal Component 2")
    
    #clustering result
    ax2.scatter(*X_test_transformed.T, c=kmeans_model.predict(X_test), s=2)
    xlims = ax2.get_xlim(); ylims = ax2.get_ylim();
    
    #------------------------------------------------------------------------------------------------------------------------------------------
    # Diagram similar to scikit learn tutorial
    # on handwritten digits data
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
    # 
    # 
    #-------------------------------------------------------------------------------------------------------------------------------------------
    vor = Voronoi(np.append(dim_reducer.transform(kmeans_model.cluster_centers_), [[999,999], [-999,999], [999,-999], [-999,-999]], axis=0))
    voronoi_plot_2d(vor, ax=ax2, show_vertices=False, line_width=3);
    for region in vor.regions:
        if -1 not in region:
            poly = [vor.vertices[i] for i in region]
            ax2.fill(*zip(*poly), alpha=0.2)

    ax2.set_xlim(xlims); ax2.set_ylim(ylims);
    ax2.set_xlabel('Principal Component 1'); ax2.set_ylabel('Principal Component 2');
    
    #save plot
    plt.savefig("result.png")
               
if __name__ == "__main__":
    main()
