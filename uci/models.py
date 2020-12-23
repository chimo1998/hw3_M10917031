import glob
from scipy.io import arff
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering as AC
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import time

class Docu:
    def __init__(self):
        self.X = None
        self.cate = None

    def load(self):
        data = arff.loadarff('%s/HTRU2/HTRU_2.arff' % os.path.dirname(__file__))
        df = pd.DataFrame(data[0])
        self.X = df.drop('class', axis=1)
        self.normalize()
        self.cate = df['class'].astype(int)
        return self.X

    def normalize(self):
        for i in self.X.columns:
            self.X[i] = (self.X[i] - self.X[i].min()) / (self.X[i].max() - self.X[i].min())

class Model:
    def __init__(self, nc=2):
        self.docu = Docu()
        self.vectors = self.docu.load()
        self.nc = nc
        self.model = None
    
    def fit(self, **args):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError

    def jodge(self):
        self.SSE()

    def SSE(self):
        sse = 0.0
        pred = self.predict()
        for i in range(self.nc):
            mask = pred == i
            x = self.vectors[mask]
            for c in self.vectors.columns:
                sse += ((x[c] - x[c].mean()) ** 2).sum()
        print("SSE : %f" % sse)
        
class DBScan(Model):
    def fit(self, eps=1.1, min_samples=7):
        scan = DBSCAN(eps = eps, min_samples = min_samples)
        t0 = time.time()
        scan.fit(self.vectors)
        print("Training cost %0.3fs" % (time.time() - t0))
        self.model = scan

    def predict(self):
        return self.model.fit_predict(self.vectors)

class KMean(Model):
    def fit(self, init='k-means++', max_iter=100, n_init=20):
        km = KMeans(n_clusters=self.nc, init=init, max_iter=max_iter, n_init=n_init)
        t0 = time.time()
        km.fit(self.vectors)
        print("Training cost %0.3fs" % (time.time() - t0))
        self.model = km

    def predict(self):
        return self.model.predict(self.vectors)

class Hierarchical(Model):
    def __init__(self):
        super().__init__()
        self.draw = None
        self.p = 5

    def fit(self, draw=False, p=5):
        self.p = p
        self.draw = draw
        hie = None
        if draw: # for drawing
            hie = AC(n_clusters=None, compute_full_tree=True, distance_threshold=0)
        else:
            hie = AC(n_clusters=self.nc)
        t0 = time.time()
        hie.fit(self.vectors)
        print("Training cost %0.3fs" % (time.time() - t0))
        self.model = hie

    def predict(self):
        return self.model.fit_predict(self.vectors)

    def dendrogram(self):
        if (not self.draw):
            print("Draw is setting to false, please init model with draw=True")
            return
        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)
        for i, merge in enumerate(self.model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        
        linkage_matrix = np.column_stack([self.model.children_, self.model.distances_,
                                          counts]).astype(float)

        dendrogram(linkage_matrix, truncate_mode='level', p=self.p)
        plt.show()
