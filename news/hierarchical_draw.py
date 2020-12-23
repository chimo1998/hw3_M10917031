import models

hie = models.Hierarchical()
hie.fit(draw=True, p=5)
hie.dendrogram()
