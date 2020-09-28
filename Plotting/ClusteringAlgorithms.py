from numpy import unique
from numpy import where
from sklearn.cluster import AffinityPropagation, KMeans, Birch, DBSCAN, MeanShift, OPTICS, SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.mixture import GaussianMixture


def cluster_with_kmeans(number_of_clusters, principal_components, principal_df):
	# do some clustering
	kmeans_pca = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
	prediction = kmeans_pca.fit_predict(principal_components)

	final_df = pd.concat([principal_df], axis=1)
	final_df['Segment'] = kmeans_pca.labels_
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)

	# plot the thing
	sns.set()
	plt.figure(figsize=(12, 7))
	plt.title("K-Means PCA")
	sns.scatterplot(
		x="PC1", y="PC2",
		hue="Segment",
		data=final_df,
		legend=False,
		s=75,
		alpha=0.7
	)

	add_race_labels(final_df)

	calc_silhouette(data=principal_components, prediction=prediction, n_clusters=number_of_clusters)

	return final_df


def calc_silhouette(data, prediction, n_clusters):
	ax = plt.gca()
	# Compute the silhouette scores for each sample
	silhouette_avg = silhouette_score(data, prediction)
	sample_silhouette_values = silhouette_samples(data, prediction)
	print(silhouette_avg)

	y_lower = padding = 2
	y_upper = 0
	for i in range(n_clusters):
		# Aggregate the silhouette scores for samples belonging to
		ith_cluster_silhouette_values = sample_silhouette_values[prediction == i]
		ith_cluster_silhouette_values.sort()

		size_cluster_i = ith_cluster_silhouette_values.shape[0]
		y_upper = y_lower + size_cluster_i

		ax.fill_betweenx(np.arange(y_lower, y_upper),
		                 0,
		                 ith_cluster_silhouette_values,
		                 alpha=0.7)

		# Label the silhouette plots with their cluster numbers at the middle
		ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))

		# Compute the new y_lower for next plot
		y_lower = y_upper + padding

	ax.set_xlabel("The silhouette coefficient values")
	ax.set_ylabel("Cluster label")

	# The vertical line for average silhouette score of all the values
	ax.axvline(x=silhouette_avg, c='r', alpha=0.8, lw=0.8, ls='-')
	ax.annotate('Average',
	            xytext=(silhouette_avg, y_lower * 1.025),
	            xy=(0, 0),
	            ha='center',
	            alpha=0.8,
	            c='r')

	ax.set_yticks([])  # Clear the yaxis labels / ticks
	ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
	ax.set_ylim(0, y_upper + 1)
	ax.set_xlim(-0.075, 1.0)
	plt.title("Silhouette Index")
	plt.show()


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
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1], s=75)
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	plt.title("Affinity Propagation")
	add_race_labels(final_df)
	calc_silhouette(data=principal_components, prediction=y_hat, n_clusters=len(clusters))
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
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1], s=75)
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	plt.title("BIRCH Clustering")
	add_race_labels(final_df)
	calc_silhouette(data=principal_components, prediction=yhat, n_clusters=len(clusters))
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
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1], s=75)
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	plt.title("DBSCAN Clustering")
	add_race_labels(final_df)
	print(final_df)
	calc_silhouette(data=principal_components, prediction=yhat, n_clusters=len(clusters))
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
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1], s=75)
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	print(final_df)
	plt.title("Mean Shift Clustering")
	add_race_labels(final_df)
	calc_silhouette(data=principal_components, prediction=yhat, n_clusters=len(clusters))
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
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1], s=75)
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	print(final_df)
	plt.title("OPTICS Clustering")
	add_race_labels(final_df)
	calc_silhouette(data=principal_components, prediction=yhat, n_clusters=len(clusters))
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
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1], s=75)
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	print(final_df)
	plt.title("Spectral clustering")
	add_race_labels(final_df)
	calc_silhouette(data=principal_components, prediction=yhat, n_clusters=len(clusters))
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
		plt.scatter(principal_components[row_ix, 0], principal_components[row_ix, 1], s=75)
	final_df.rename({0: 'PC1', 1: 'PC2', 2: 'PC3', 'y': 'Race'}, axis=1, inplace=True)
	print(final_df)
	plt.title("Gaussian Clustering")
	add_race_labels(final_df)
	calc_silhouette(data=principal_components, prediction=yhat, n_clusters=len(clusters))
	return final_df


def add_race_labels(data):
	if len(data) > 100:
		plt.show()
	else:
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
