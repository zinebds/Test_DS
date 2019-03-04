# Test_DS
Le langage utilisé : Pyspark

Etapes suivies :

  1. Chargement des données et prise en main des variables,la variable "Number" a été considérée comme le nombre de vélos dans chaque station.
  2. Centrer réduire les données (Longitude,Latitude,Number)
  3. Appliquer la PCA (analyse en composante principale) pour permettre une meilleur visualisation des résultats sou format en 2 dimensions après clustering.
  La variance exoliquée sur les deux premiers axes est de 90%
  4. Applique l'algorithme Kmeans, pour une classification non supervisée, après analyse des coùts et vu la taille des données, le nombre de clusters a été fixé à 4.
  5. Le Plot des résultats est en commentaire.
  6. Sauvegarde des données
  

## Usage
le premier argument script.py est le code pyspark.
Le second correnspond à l'emplacement des données.
Le dernier correspond à l'emplacement où vous voulez enregistrer les données.

     spark-submit script.py Brisbane_CityBike.json results.csv
