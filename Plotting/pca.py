import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.colors as mcolors


def main():
	raw_data = pd.read_csv("data_file.csv")
	grouped_by_race = raw_data.groupby(["Race"]).count().reset_index()
	print(grouped_by_race["Race"].describe())
	print(raw_data["Race"].describe())

	features = ['Value', 'Passes', 'MetersPassed', 'MetersRun', 'Touchdowns', 'Score', 'Blocks', 'Breaks', 'Knockouts',
	            'Stuns', 'Casualties', 'Kills', 'Catches', 'Interceptions', 'Dodges', 'GFIs', 'Pickups', 'BallPossession',
	            'Completions', 'Sacks', 'Turnovers']
	x = raw_data.loc[:, features].values
	y = raw_data.loc[:, ['Race']].values
	x = StandardScaler().fit_transform(x)
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(x)
	principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
	finalDf = pd.concat([principalDf, raw_data[['Race']]], axis=1)

	# fig = plt.figure(figsize=(8, 8))
	# ax = fig.add_subplot(1, 1, 1)
	# ax.set_xlabel('PC1', fontsize=15)
	# ax.set_ylabel('PC2', fontsize=15)
	# ax.set_title('PCA', fontsize=20)

	finalDf['y'] = y
	plt.figure(figsize=(16, 10))
	sns.scatterplot(
		x="PC1", y="PC2",
		hue="y",
		palette=sns.color_palette("hls", 24),
		data=finalDf,
		legend="brief",
		alpha=0.3
	)

	# targets = grouped_by_race["Race"].values
	# colors = mcolors.XKCD_COLORS
	# for target, color in zip(targets, colors):
	# 	indicesToKeep = finalDf['Race'] == target
	# 	ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
	# 	           , finalDf.loc[indicesToKeep, 'PC2']
	# 	           , c=color
	# 	           , s=30)
	# ax.legend(targets)
	# ax.grid()

	plt.show()


if __name__ == '__main__':
	main()
