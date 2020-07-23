import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc


def main():
	raw_data = pd.read_csv("data_file.csv")
	grouped_by_race = raw_data.groupby(["Race"]).count().reset_index()
	races = grouped_by_race.loc[:, ['Race']].values
	print(grouped_by_race.head())
	grouped_by_race = grouped_by_race.drop("Race", axis=1)

	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(grouped_by_race.iloc[:, :-1])
	x = StandardScaler().fit_transform(principalComponents)
	principal_Df = pd.DataFrame(data=x, columns=['PC1', 'PC2'])
	principal_Df['y'] = races

	# Find the number of clusters, plot twist its 4
	wcss = []
	for i in range(1, 14):
		kmeans_pca = KMeans(n_clusters=i, init='k-means++')
		kmeans_pca.fit(x)
		wcss.append(kmeans_pca.inertia_)

	# plt.figure(figsize=(16, 10))
	# plt.plot(range(1, 14), wcss, marker='o', linestyle='--')
	# plt.xlabel("Number of clusters")
	# plt.ylabel("WCSS")
	# plt.title("K-Means PCA2")
	# plt.show()

	# Dendrogram
	# plt.figure(figsize=(10, 7))
	# plt.title("Race Dendograms")
	# dend = shc.dendrogram(shc.linkage(x, method='ward'), orientation="right", labels=races)
	# plt.show()

	# plt.figure(figsize=(16, 10))
	# sns.scatterplot(
	# 	x="PC1", y="PC2",
	# 	hue="y",
	# 	palette=sns.color_palette("hls", 24),
	# 	data=principal_Df,
	# 	legend="full",
	# 	alpha=1
	# )
	# plt.show()

	number_of_clusters = 4

	# do some clustering
	kmeans_pca = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
	kmeans_pca.fit(x)

	principalDf = pd.DataFrame(data=principal_Df, columns=['PC1', 'PC2'])
	finalDf = pd.concat([principalDf, principal_Df['y']], axis=1)
	finalDf['Segment'] = kmeans_pca.labels_
	finalDf['Races'] = races

	# Extract clusters
	cluster1 = finalDf.loc[finalDf.Segment == 1]
	cluster2 = finalDf.loc[finalDf.Segment == 2]
	cluster0 = finalDf.loc[finalDf.Segment == 0]
	print(cluster1)

	# plot the thing
	sns.set()
	plt.figure(figsize=(16, 10))
	plt.title("K-Means PCA PCA2")
	sns.scatterplot(
		x="PC1", y="PC2",
		hue="Races",
		# palette=sns.color_palette("hls", 24),
		data=finalDf,
		style=finalDf["Segment"],
		legend="brief",
		alpha=0.7
	)
	plt.show()

	# Add cluster number to the original dataset for future use
	# grouped_by_race['Segment'] = kmeans_pca.labels_
	# grouped_by_race['Race'] = races
	# print(grouped_by_race.describe())
	# grouped_by_race.to_csv("PCA2_Grouped_By_Race_With_Cluster_Number.csv")


if __name__ == '__main__':
	main()
