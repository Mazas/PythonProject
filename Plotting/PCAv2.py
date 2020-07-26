import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc

from kneed import KneeLocator


def save_clusters(grouped_by_race, kmeans_pca, races):
	# Add cluster number to the original data set for future use
	grouped_by_race['Segment'] = kmeans_pca.labels_
	grouped_by_race['Race'] = races
	print(grouped_by_race.describe())
	grouped_by_race.to_csv("PCA2_Grouped_By_Race_With_Cluster_Number.csv")


def plot_3d_scatter(final_df):
	# 3D plot
	sns.set_style("whitegrid", {'axes.grid': False})
	xs = final_df.loc[:, ['PC1']].values
	ys = final_df.loc[:, ['PC2']].values
	zs = final_df.loc[:, ['PC3']].values
	figure = plt.figure(figsize=(6, 6))
	plt.title("K-Means PCA")
	ax = Axes3D(figure)
	fig = ax.scatter(xs, ys, zs, s=50, c=final_df.loc[:, ['Segment']], marker='o', depthshade=False, cmap='Paired')

	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')
	ax.set_zlabel('PC3')

	legend = ax.legend(*fig.legend_elements(), loc="lower center", title="X Values", borderaxespad=-10, ncol=4)
	ax.add_artist(legend)

	xAxisLine = ((min(final_df['PC1']), max(final_df['PC1'])), (0, 0), (0, 0))
	ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
	yAxisLine = ((0, 0), (min(final_df['PC2']), max(final_df['PC2'])), (0, 0))
	ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
	zAxisLine = ((0, 0), (0, 0), (min(final_df['PC3']), max(final_df['PC3'])))
	ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

	plt.show()


def find_number_of_components(pca):
	# find number of components
	# 36 components account for 98.7 variance
	# meaning that last 11 components are close to useless
	per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
	ratio_sum = 0
	for order in range(0, len(pca.explained_variance_ratio_)):
		ratio_sum = ratio_sum + pca.explained_variance_ratio_[order]
		print("{0}: {1}".format(order, ratio_sum))

	labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
	plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
	plt.ylabel("Percentage of Explained Variance")
	plt.xlabel("Principal Component")
	plt.title("Explained variance")
	plt.show()


def find_the_number_of_clusters(principal_components, limit):
	# Find the number of clusters
	wcss = []
	for i in range(1, limit + 1):
		print("Fitting components {0}/{1}".format(i, limit))
		kmeans_pca = KMeans(n_clusters=i, init='k-means++')
		kmeans_pca.fit(principal_components)
		wcss.append(kmeans_pca.inertia_)

	# Plot the figure
	plt.figure(figsize=(16, 10))
	plt.plot(range(1, limit + 1), wcss, marker='o', linestyle='--')
	plt.xlabel("Number of clusters")
	plt.ylabel("WCSS")
	plt.title("K-Means PCA2")

	order = np.linspace(1, limit, limit)
	# find the elbow
	# https://github.com/arvkevi/kneed/blob/master/notebooks/decreasing_function_walkthrough.ipynb
	kn = KneeLocator(order, wcss, curve='convex', direction='decreasing')
	plt.vlines(kn.elbow, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
	plt.show()
	print("Number of clusters: {0}".format(kn.elbow))
	return int(kn.elbow)


def plot_dendrogram(races, x):
	# Dendrogram
	plt.figure(figsize=(10, 7))
	plt.title("Race Dendograms")
	shc.dendrogram(shc.linkage(x, method='ward'), orientation="right", labels=races)
	plt.show()


def cluster_with_kmeans(number_of_clusters, principal_components, principal_df, races):
	# do some clustering
	kmeans_pca = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
	kmeans_pca.fit(principal_components)

	finalDf = pd.concat([principal_df], axis=1)
	finalDf['Segment'] = kmeans_pca.labels_
	finalDf['Race'] = races
	finalDf.rename({0: 'PC1', 1: 'PC2', 2: 'PC3'}, axis=1, inplace=True)
	grouped = finalDf.groupby(["Race"]).median().reset_index()

	# plot the thing
	sns.set()
	plt.figure(figsize=(20, 20))
	plt.title("K-Means PCA")
	sns.scatterplot(
		x="PC1", y="PC2",
		hue="Segment",
		data=grouped,
		style=grouped["Segment"],
		legend=False,
		alpha=0.7
	)

	# add labels next to the data point
	for line in range(0, grouped.shape[0]):
		plt.text(
			grouped.PC1[line] + 0.05,
			grouped.PC2[line] - 0.1,
			grouped.Race[line],
			horizontalalignment='left',
			size='small',
			color='black'
		)

	plt.show()
	return finalDf


def main():
	# load data into a data frame
	raw_data = pd.read_csv("data_file.csv")

	# data frame that is grouped by race
	# used for dendrogram
	grouped_by_race = raw_data.groupby(["Race"]).median().reset_index()
	distinct_races = grouped_by_race.loc[:, ['Race']].values

	# extract races from the data frame into separate dictionary
	# then scale/ normalize remaining data
	races = raw_data.loc[:, ['Race']].values
	full_df = raw_data.drop("Race", axis=1)
	full_df = StandardScaler().fit_transform(full_df)

	# fit principal components
	# store them into separate data frame
	# add labels for race to the data frame
	number_of_components = 37
	pca = PCA(n_components=number_of_components)
	principal_components = pca.fit_transform(full_df)
	principal_Df = pd.DataFrame(data=principal_components)
	principal_Df['y'] = races

	# number of clusters for k-means clustering algorithm


	# to plot dendrogram, data set must be small
	# otherwise it runs out of memory
	# this makes dendrogram quite inconsistent and frankly useless
	# can be somewhat fixed by using pca grouped pca
	plot_dendrogram(distinct_races, x=grouped_by_race.drop("Race", axis=1))

	# find number of components
	find_number_of_components(pca)

	# find number of clusters
	number_of_clusters = find_the_number_of_clusters(principal_components, 20)

	# Cluster with k-means
	# returns final_df which is a data frame with cluster numbers and is grouped by race
	final_df = cluster_with_kmeans(number_of_clusters, principal_components, principal_Df, races)

	raw_data['Segment'] = final_df.Segment
	print(raw_data.Segment.describe())
	raw_data.to_csv("PCA2_With_Cluster_Number_Raw_Data.csv")

	# plot the final df in 3D scatter graph
	plot_3d_scatter(final_df.groupby(["Race"]).median().reset_index())


if __name__ == '__main__':
	main()
