import models

scan = models.DBScan()
scan.fit()
print("Purity %f" % scan.get_purity())
