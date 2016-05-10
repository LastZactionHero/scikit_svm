import csv
import code
from sklearn.cluster import DBSCAN
from sklearn import cluster, datasets
from matplotlib import pyplot
import numpy as np
from sklearn import preprocessing

# filename_single_family = "./data/Zip_ZRI_SingleFamilyResidenceRental.csv"
filename_single_family = "./data/Zip_Zri_SingleFamilyResidenceRental.csv"

labels_single_family = []
data_single_family = []

with open(filename_single_family) as csvfile:
    reader = csv.DictReader(csvfile)
    # selected_fields = ['2013-10', '2013-11', '2013-12', '2014-01', '2014-02', '2014-03', '2014-04', '2014-05', '2014-06', '2014-07', '2014-08', '2014-09', '2014-10', '2014-11', '2014-12', '2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12', '2016-01', '2016-02', '2016-03']
    selected_fields = ["2013-01","2013-02","2013-03","2013-04","2013-05","2013-06","2013-07","2013-08","2013-09","2013-10","2013-11","2013-12","2014-01","2014-02","2014-03","2014-04","2014-05","2014-06","2014-07","2014-08","2014-09","2014-10","2014-11","2014-12","2015-01","2015-02","2015-03","2015-04","2015-05","2015-06","2015-07","2015-08","2015-09","2015-10","2015-11","2015-12","2016-01","2016-02","2016-03"]

    for row in reader:
        if row['State'] != 'CO':
            continue

        parsed = []
        for field in selected_fields:
            if len(row[field]) > 0:
                parsed.append(float(row[field]))
            else:
                parsed.append(0)

        data_single_family.append(parsed)
        labels_single_family.append([row['RegionName'], row['City'], row['State']])

# code.interact(local=locals())

data_single_family = preprocessing.scale(data_single_family)

cluster_count = 20
k_means = cluster.KMeans(n_clusters=cluster_count)
k_means.fit(data_single_family)

labels = k_means.labels_
centroids = k_means.cluster_centers_

# code.interact(local=locals())
clusters=[[] for i in range(cluster_count)]

for i in range(len(k_means.labels_)):
    cluster_id = k_means.labels_[i]
    label = labels_single_family[i]
    clusters[cluster_id].append(label)

for cluster in clusters:
    print("*******************")
    for label in cluster:
        print(label)
#     
# for i in range(cluster_count):
#     print("Cluster %s" % (i))
#     cluster_indices = np.where(labels==i)[0]
#     for cluster_idx in cluster_indices:
#         print(labels_single_family[cluster_idx])