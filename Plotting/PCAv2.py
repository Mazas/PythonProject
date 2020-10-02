import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import cdist
import Plotting.ClusteringAlgorithms as clustering
from kneed import KneeLocator


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

    xAxisLine = ((min(final_df['PC1']), max(final_df['PC1'])),
                 ((min(final_df['PC2']) + max(final_df['PC2'])) / 2, (min(final_df['PC2']) + max(final_df['PC2'])) / 2),
                 ((min(final_df['PC3']) + max(final_df['PC3'])) / 2, (min(final_df['PC3']) + max(final_df['PC3'])) / 2))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = (((min(final_df['PC1']) + max(final_df['PC1'])) / 2, (min(final_df['PC1']) + max(final_df['PC1'])) / 2),
                 (min(final_df['PC2']), max(final_df['PC2'])),
                 ((min(final_df['PC3']) + max(final_df['PC3'])) / 2, (min(final_df['PC3']) + max(final_df['PC3'])) / 2))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = (((min(final_df['PC1']) + max(final_df['PC1'])) / 2, (min(final_df['PC1']) + max(final_df['PC1'])) / 2),
                 ((min(final_df['PC2']) + max(final_df['PC2'])) / 2, (min(final_df['PC2']) + max(final_df['PC2'])) / 2),
                 (min(final_df['PC3']), max(final_df['PC3'])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

    plt.show()


def find_number_of_components(pca):
    # find number of components
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
        kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans_pca.fit(principal_components)
        wcss.append(sum(np.min(cdist(principal_components, kmeans_pca.cluster_centers_, 'euclidean'),
                               axis=1)) / principal_components.shape[0])

    # Plot the figure
    plt.figure(figsize=(16, 10))
    plt.plot(range(1, limit + 1), wcss, marker='o', linestyle='--')
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.title("K-Means PCA")

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


def drop_columns(data):
    # ['Race', 'Passes', 'MetersRun', 'Touchdowns', 'Score', 'Blocks',
    # 'Breaks', 'Knockouts', 'Stuns', 'Casualties', 'Kills', 'Catches',
    # 'CatchesFailed', 'Interceptions', 'Dodges', 'DodgesFailed', 'GFIs',
    # 'Pickups', 'PickupsFailed', 'BallPossession', 'Completions', 'Sacks',
    # 'Turnovers', 'XP']
    # has the most effect on the clustering.
    # note that K means clusters this subset the same way as full feature set
    # however hierarchical clustering still finds only 2 clusters

    columns = ['Value', 'IsHome', 'BlocksFailed', 'BlocksNeutral', 'BlocksGood', 'BlockRisk', 'BlockLuck',
               'MoveRisk', 'MoveLuck', 'PassRisk', 'PassLuck', 'Risk', 'Luck', 'GFIsFailed', 'PassesFailed']
    return data.drop(columns, axis=1)


def main():
    # load data into a data frame
    raw_data = pd.read_csv("data_file.csv")
    raw_data = drop_columns(raw_data)

    print(raw_data.columns)
    print(len(raw_data.columns))

    # data frame that is grouped by race
    # used for dendrogram
    # grouped_by_race = raw_data.groupby(["Race"]).median().reset_index()
    # distinct_races = grouped_by_race.loc[:, ['Race']].values

    # extract races from the data frame into separate dictionary
    # then scale/ normalize remaining data
    original_groups = raw_data.copy()
    races = raw_data.loc[:, ['Race']].values
    full_df = raw_data.drop("Race", axis=1)
    full_df = StandardScaler().fit_transform(full_df)

    number_of_components = 27
    pca = PCA(n_components=number_of_components)
    principal_components = pca.fit_transform(full_df)
    principal_Df = pd.DataFrame(data=principal_components)
    principal_Df['y'] = races

    # original_groups = grouped_by_race.copy()
    #
    # grouped_by_race = grouped_by_race.drop("Race", axis=1)
    # grouped_by_race = StandardScaler().fit_transform(grouped_by_race)
    #
    # # fit principal components
    # # store them into separate data frame
    # # add labels for race to the data frame
    # number_of_components = 15
    # pca = PCA(n_components=number_of_components)
    # principal_components = pca.fit_transform(grouped_by_race)
    # principal_Df = pd.DataFrame(data=principal_components)
    # principal_Df['y'] = distinct_races

    # to plot dendrogram, data set must be small
    # otherwise it runs out of memory
    # this makes dendrogram quite inconsistent and frankly useless
    # can be somewhat fixed by using pca grouped pca
    #plot_dendrogram(distinct_races, x=grouped_by_race)

    # find number of components
    # find_number_of_components(pca)

    # find number of clusters
    number_of_clusters = find_the_number_of_clusters(principal_components, 10)

    # Cluster with k-means
    # returns final_df which is a data frame with cluster numbers and is grouped by race
    final_df = clustering.cluster_with_kmeans(number_of_clusters, principal_components, principal_Df)

    # cluster with affinity propagation
    #clustering.affinity_propagation(principal_components, principal_Df)

    # Balanced Iterative Reducing and Clustering using Hierarchies
    # BIRCH clustering
    #clustering.birch_clustering(principal_components, principal_Df, number_of_clusters)

    # DBSCAN clustering
    #clustering.dbscan_clustering(principal_components, principal_Df)

    # Mean Shift clustering
    #clustering.mean_shift_clustering(principal_components, principal_Df)

    # OPTICS clustering, modified DBSCAN
    #clustering.optics_clustering(principal_components, principal_Df)

    # spectral clustering
    #clustering.spectral_clustering(principal_components, principal_Df)

    # gaussian clustering
    #clustering.gaussian_clustering(principal_components, principal_Df)

    # plot the final df in 3D scatter graph
    # plot_3d_scatter(final_df)

    original_groups['Segment'] = final_df['Segment']
    # print(original_groups)

    # save to a file
    original_groups.to_csv("raw_with_cluster_labels.csv")


if __name__ == '__main__':
    main()
