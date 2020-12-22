import models

hie = models.Hierarchical()
hie.fit()
print("Purity : %f" % hie.get_purity())
