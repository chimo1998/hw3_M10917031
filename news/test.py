import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import time

class docu:
    def __init__(self):
        self.text = []
        self.cate = []
    def append(self, text, cate):
        self.text.append(text)
        self.cate.append(cate)

kn = 20
folder_name = "news/mini_newsgroups"
folders = os.listdir(folder_name)
data = docu()

for n in range(len(folders)):
    files = glob.glob(("%s/%s/*" % (folder_name, folders[n])))
    for f in files:
        with open(f, "r", encoding="utf-8", errors='ignore') as fin:
            data.append(fin.read(),n)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data.text)

km = KMeans(n_clusters=kn, init='k-means++', max_iter=30, n_init=1)
t0 = time.time()
km.fit(vectors)
print("time : %0.3fs" % (time.time() - t0))
print(km.predict(vectors))
