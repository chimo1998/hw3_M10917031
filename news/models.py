import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering as AC
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import time

class Docu:
    def __init__(self):
        self.text = []
        self.cate = []

    def append(self, text, cate):
        self.text.append(text)
        self.cate.append(cate)

    def load(self):
        kn = 20
        folder_name = "mini_newsgroups"
        folders = os.listdir(os.path.join(os.getcwd(), folder_name))
        data = Docu()

        for n in range(len(folders)):
            files = glob.glob(("%s/%s/*" % (folder_name, folders[n])))
            for f in files:
                with open(f, "r", encoding="utf-8", errors='ignore') as fin:
                    data.append(fin.read(),n)

        return data

class Model:
    def __init__(self, nc=20, max_df=0.3, min_df=0.035):
        self.docu = Docu().load()
        vectorizer = TV(max_df=max_df, min_df=min_df)
        self.vectors = vectorizer.fit_transform(self.docu.text).todense()
        self.nc = nc
        self.model = None
    
    def fit(self, **args):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError

    def get_purity(self, *args):
        count = [[0] * self.nc for i in range(self.nc)]
        pred = self.predict()

        for index in range(len(pred)):
            count[pred[index]][self.docu.cate[index]] += 1
        purity = 0.0

        for i in range(self.nc):
            purity += (max(count[i]) / len(pred))

        return purity
        
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

    def fit(self, draw=False):
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

        dendrogram(linkage_matrix, truncate_mode='level', p=4)
        plt.show()
