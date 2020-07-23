import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc


def main():
	raw_data = pd.read_csv("data_file.csv")
	# grouped_by_race = raw_data.groupby(["Race"]).count().reset_index()
	races = raw_data.loc[:, ['Race']].values
	grouped_by_race = raw_data.drop("Race", axis=1)
	grouped_by_race = StandardScaler().fit_transform(grouped_by_race)

	pca = PCA(n_components=36)
	principalComponents = pca.fit_transform(grouped_by_race)

	# find number of components
	# 36 components account for 99 variance
	# meaning that last 11 components are close to useless
	# per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
	# ratio_sum = 0
	# for order in range(0, len(pca.explained_variance_ratio_)):
	# 	ratio_sum = ratio_sum + pca.explained_variance_ratio_[order]
	# 	print("{0}: {1}".format(order, ratio_sum))
	#
	# labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
	# plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
	# plt.ylabel("Percentage of Explained Variance")
	# plt.xlabel("Principal Component")
	# plt.title("Explained variance")
	# plt.show()

	principal_Df = pd.DataFrame(data=principalComponents)
	principal_Df['y'] = races

	# Find the number of clusters
	# wcss = []
	# for i in range(1, 10):
	# 	print("Fitting components {0}/36".format(i))
	# 	kmeans_pca = KMeans(n_clusters=i, init='k-means++')
	# 	kmeans_pca.fit(principalComponents)
	# 	wcss.append(kmeans_pca.inertia_)
	#
	# plt.figure(figsize=(16, 10))
	# plt.plot(range(1, 10), wcss, marker='o', linestyle='--')
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
	kmeans_pca.fit(principalComponents)

	finalDf = pd.concat([principal_Df], axis=1)
	finalDf['Segment'] = kmeans_pca.labels_
	finalDf['Race'] = races
	finalDf.rename({0: 'PC1', 1: 'PC2', 2: 'PC3'}, axis=1, inplace=True)
	finalDf = finalDf.groupby(["Race"]).first().reset_index()

	# Extract clusters
	# cluster1 = finalDf.loc[finalDf.Segment == 1]
	# cluster2 = finalDf.loc[finalDf.Segment == 2]
	# cluster0 = finalDf.loc[finalDf.Segment == 0]
	# cluster3 = finalDf.loc[finalDf.Segment == 3]
	print(finalDf['Segment'].values)

	# plot the thing
	sns.set()
	plt.figure(figsize=(20, 20))
	plt.title("K-Means PCA")
	sns.scatterplot(
		x="PC1", y="PC2",
		hue="Segment",
		data=finalDf,
		style=finalDf["Segment"],
		legend=False,
		alpha=0.7
	)

	for line in range(0, finalDf.shape[0]):
		plt.text(
			finalDf.PC1[line] + 0.05,
			finalDf.PC2[line]-0.1,
			finalDf.Race[line],
			horizontalalignment='left',
			size='small',
		    color='black'
		)

	plt.show()

	# 3D plot
	# sns.set_style("whitegrid", {'axes.grid': False})
	# xs = finalDf.loc[:, ['PC1']].values
	# ys = finalDf.loc[:, ['PC2']].values
	# zs = finalDf.loc[:, ['PC3']].values
	# figure = plt.figure(figsize=(6, 6))
	# plt.title("K-Means PCA")
	# ax = Axes3D(figure)
	# fig = ax.scatter(xs, ys, zs, s=50, c=finalDf.loc[:, ['Segment']], marker='o', depthshade=False, cmap='Paired')
	#
	# ax.set_xlabel('PC1')
	# ax.set_ylabel('PC2')
	# ax.set_zlabel('PC3')
	#
	# legend = ax.legend(*fig.legend_elements(), loc="lower center", title="X Values", borderaxespad=-10, ncol=4)
	# ax.add_artist(legend)
	#
	# xAxisLine = ((min(finalDf['PC1']), max(finalDf['PC1'])), (0, 0), (0, 0))
	# ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
	# yAxisLine = ((0, 0), (min(finalDf['PC2']), max(finalDf['PC2'])), (0, 0))
	# ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
	# zAxisLine = ((0, 0), (0, 0), (min(finalDf['PC3']), max(finalDf['PC3'])))
	# ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
	#
	# plt.show()


	# Add cluster number to the original dataset for future use
	# grouped_by_race['Segment'] = kmeans_pca.labels_
	# grouped_by_race['Race'] = races
	# print(grouped_by_race.describe())
	# grouped_by_race.to_csv("PCA2_Grouped_By_Race_With_Cluster_Number.csv")


if __name__ == '__main__':
	main()
