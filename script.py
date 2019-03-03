# Creation du sparkContext 
import sys
import pyspark
from pyspark.sql import SQLContext
import logging

# Validaiton des arguments
arguments = sys.argv
if (len(arguments)<3):
	logging.error("le nombre de paramétres est insuffisant")
	logging.error("merci de respecter le format suivant: COMMAND fichier_de_donnees fichier_resultat")
	exit()

path_to_data = arguments[1]
path_to_save_data =arguments[2]

sc = pyspark.SparkContext('local[*]')
spark = SQLContext(sc)
data = spark.read.format('json').option('header','true').load(path_to_data)

# Réduction du bruit de log
sc.setLogLevel("ERROR")

# Import libraries
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans,KMeansModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
from pyspark.ml.feature import StandardScaler

# Scaler Data and apply PCA

pick = ["latitude","longitude",'number']
assembler = VectorAssembler(inputCols = pick,outputCol = 'features')
test = assembler.transform(data).select('name','features')
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)

scalerModel = scaler.fit(test)
scaledData = scalerModel.transform(test)
pca = PCA(k = 3,inputCol = "scaledFeatures", outputCol = "pcaFeatures")
model = pca.fit(scaledData)
df_pca = model.transform(scaledData).select('name','pcaFeatures')

model.explainedVariance #verify efficiency of PCA, ~ 90% for two first component

# find the right number of clusters 
cost = np.zeros(10)
for k in range(2,10):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("pcaFeatures")
    model = kmeans.fit(df_pca)
    cost[k] = model.computeCost(df_pca) # requires Spark 2.0 or later
#fig, ax = plt.subplots(1,1, figsize =(8,6))
#ax.plot(range(2,10),cost[2:10])
#ax.set_xlabel('k')
#ax.set_ylabel('cost')

# We choose to predict 4 clusters
k = 4
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("pcaFeatures")
model_km = kmeans.fit(df_pca)
centers = model.clusterCenters()
#We can save the model and only run prediction later:
#model_km.save(path_to_save_model)


transformed = model_km.transform(df_pca).select('name','prediction')
rows = transformed.collect()

#Join stations characteristics with cluster predictions
df_pred = spark.createDataFrame(rows)
df_pred = df_pred.join(df_pca, 'name')
real_data = data.join(df_pred,'name')

# plot the result to see how efficient is our clustering
z_df = df_pred.select('pcaFeatures').rdd.map(lambda x: x[0].toArray().tolist()).toDF()
df_pred =df_pred.drop('pcaFeatures')
from pyspark.sql.types import LongType
def zipindexdf(pca_df):
    """
    :param pca_df: spark dataframe to which an index column is to be added
    :return: same dataframe but with an additional index column
    """
    schema_new = pca_df.schema.add("index", LongType(), False)
    return pca_df.rdd.zipWithIndex().map(lambda l: list(l[0]) + [l[1]]).toDF(schema_new)

df_index = zipindexdf(df_pred)
z_index = zipindexdf(z_df)
df_new =z_index .join(df_index, "index", "inner")
df_new = df_new.drop('index')

to_plot = df_new.toPandas()

## 2D
#fig, ax = plt.subplots()
#colors = {'D':'red', 'E':'blue', 'F':'green', 'G':'black'}
#ax.scatter(to_plot['_1'], to_plot['_2'], c=to_plot['prediction'])
#plt.show()

## 3D
#threedee = plt.figure(figsize=(12,10)).gca(projection='3d')
#threedee.scatter(to_plot['_1'], to_plot['_2'], to_plot['_3'], c=to_plot['prediction'])
#threedee.set_xlabel('x')
#threedee.set_ylabel('y')
#threedee.set_zlabel('z')
#plt.show()

#Save data
real_data.select('name','address','latitude','longitude','number','prediction').repartition(1).write.format("csv").save(path_to_save_data,header = 'true')

sc.stop()
