import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors


def main():
	raw_data = pd.read_csv("data_file.csv")
	grouped_by_race = raw_data.groupby(["Race"]).count().reset_index()

	features = ['Value', 'Passes', 'MetersPassed', 'MetersRun', 'Touchdowns', 'Score', 'Blocks', 'Breaks', 'Knockouts',
	            'Stuns', 'Casualties', 'Kills', 'Catches', 'Interceptions', 'Dodges', 'GFIs', 'Pickups',
	            'BallPossession',
	            'Completions', 'Sacks', 'Turnovers']
	x = grouped_by_race.loc[:, features].values
	y = grouped_by_race.loc[:, ['Race']].values
	x = StandardScaler().fit_transform(x)
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(x)

	# Find the number of clusters, plot twist its 3
	# wcss = []
	# for i in range(1, 14):
	# 	kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
	# 	kmeans_pca.fit(principalComponents)
	# 	wcss.append(kmeans_pca.inertia_)
	#
	# plt.figure(figsize=(16, 10))
	# plt.plot(range(1, 14), wcss, marker='o', linestyle='--')
	# plt.xlabel("Number of clusters")
	# plt.ylabel("WCSS")
	# plt.title("K-Means PCA")
	# plt.show()

	number_of_clusters = 3

	# do some clustering
	kmeans_pca = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
	kmeans_pca.fit(principalComponents)

	principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
	finalDf = pd.concat([principalDf, grouped_by_race[['Race']]], axis=1)
	finalDf['Segment'] = kmeans_pca.labels_
	finalDf['Races'] = y

	# Extract clusters
	cluster1 = finalDf.loc[finalDf.Segment == 1]
	cluster2 = finalDf.loc[finalDf.Segment == 2]
	cluster0 = finalDf.loc[finalDf.Segment == 0]
	print(cluster1)

	# plot the thing
	# sns.set()
	# plt.figure(figsize=(16, 10))
	# sns.scatterplot(
	# 	x="PC1", y="PC2",
	# 	hue="Races",
	# 	# palette=sns.color_palette("hls", 24),
	# 	data=cluster0,
	# 	style=finalDf["Segment"],
	# 	legend="brief",
	# 	alpha=0.7
	# )
	# plt.show()

	# Add cluster number to the original dataset for future use
	grouped_by_race['Segment'] = kmeans_pca.labels_
	print(grouped_by_race.describe())
	grouped_by_race.to_csv("Grouped_By_Race_With_Cluster_Number.csv")


if __name__ == '__main__':
	main()
