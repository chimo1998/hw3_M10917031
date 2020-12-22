import models

km = models.KMean()
km.fit()
print("Purity : %f" % km.get_purity())
