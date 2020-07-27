from numpy import unique
from numpy import where
from sklearn.cluster import AffinityPropagation, KMeans, Birch, DBSCAN, MeanShift, OPTICS, SpectralClustering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.mixture import GaussianMixture


def cluster_with_kmeans(number_of_clusters, principal_components, principal_df):
	# do some clustering
	kmeans_pca = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
	kmeans_pca.fit(principal_components)

	final_df = pd.concat([principal_df], axis=1)
	final_df['Segment'] = kmeans_pca.labels_
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)

	# plot the thing
	sns.set()
	plt.figure(figsize=(20, 20))
	plt.title("K-Means PCA")
	sns.scatterplot(
		x="PC1", y="PC2",
		hue="Segment",
		data=final_df,
		style=final_df["Segment"],
		legend=False,
		alpha=0.7
	)
	add_race_labels(final_df)
	return final_df


def affinity_propagation(principal_components, principal_df):
	final_df = pd.concat([principal_df], axis=1)
	model = AffinityPropagation(damping=0.9, random_state=0)
	# fit the model
	model.fit(principal_components)
	# assign a cluster to each example
	y_hat = model.predict(principal_components)
	# retrieve unique clusters
	clusters = unique(y_hat)
	final_df['Segment'] = model.labels_
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(y_hat == cluster)
		# create scatter of these samples
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1])
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	add_race_labels(final_df)
	return final_df


def birch_clustering(principal_components, principal_df, number_of_clusters):
	final_df = pd.concat([principal_df], axis=1)
	model = Birch(threshold=0.01, n_clusters=number_of_clusters)
	# fit the model
	model.fit(principal_components)
	# assign a cluster to each example
	yhat = model.predict(principal_components)
	# retrieve unique clusters
	clusters = unique(yhat)
	final_df['Segment'] = model.labels_
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1])
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	add_race_labels(final_df)
	return final_df


def dbscan_clustering(principal_components, principal_df):
	final_df = pd.concat([principal_df], axis=1)

	model = DBSCAN(eps=4, min_samples=2)
	# fit model and predict clusters
	yhat = model.fit_predict(principal_components)
	# retrieve unique clusters
	clusters = unique(yhat)
	final_df['Segment'] = model.labels_
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1])
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	add_race_labels(final_df)
	return final_df


def mean_shift_clustering(principal_components, principal_df):
	final_df = pd.concat([principal_df], axis=1)
	model = MeanShift()
	# fit model and predict clusters
	yhat = model.fit_predict(principal_components)
	# retrieve unique clusters
	clusters = unique(yhat)
	final_df['Segment'] = model.labels_
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1])
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	print(final_df)
	add_race_labels(final_df)
	return final_df


def optics_clustering(principal_components, principal_df):
	final_df = pd.concat([principal_df], axis=1)
	model = OPTICS(eps=5, min_samples=2)
	# fit model and predict clusters
	yhat = model.fit_predict(principal_components)
	# retrieve unique clusters
	clusters = unique(yhat)
	final_df['Segment'] = model.labels_
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1])
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	print(final_df)
	add_race_labels(final_df)
	return final_df


def spectral_clustering(principal_components, principal_df):
	final_df = pd.concat([principal_df], axis=1)
	model = SpectralClustering(n_clusters=3)
	# fit model and predict clusters
	yhat = model.fit_predict(principal_components)
	# retrieve unique clusters
	clusters = unique(yhat)
	final_df['Segment'] = model.labels_
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1])
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	print(final_df)
	add_race_labels(final_df)
	return final_df


def gaussian_clustering(principal_components, principal_df):
	final_df = pd.concat([principal_df], axis=1)
	model = GaussianMixture(n_components=5)
	# fit model and predict clusters
	yhat = model.fit_predict(principal_components)
	# retrieve unique clusters
	clusters = unique(yhat)
	final_df['Segment'] = model.covariance_type
	# create scatter plot for samples from each cluster
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = where(yhat == cluster)
		# create scatter of these samples
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1])
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	print(final_df)
	add_race_labels(final_df)
	return final_df


def add_race_labels(data):
	# add labels next to the data point
	for line in range(0, data.shape[0]):
		plt.text(
			data.PC1[line] + 0.05,
			data.PC2[line] - 0.1,
			data.Race[line],
			horizontalalignment='left',
			size='small',
			color='black'
		)
	# show the plot
	plt.show()


