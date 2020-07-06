import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
import seaborn as sns


def main():
	raw_data = pd.read_csv("data_file.csv")
	df_cifar = raw_data.groupby(["Race"]).count().reset_index()
	races = df_cifar["Race"].values
	print(df_cifar.head())
	df_cifar = df_cifar.drop("Race", axis=1)

	pca_cifar = PCA(n_components=2)
	principalComponents_cifar = pca_cifar.fit_transform(df_cifar.iloc[:, :-1])

	principal_cifar_Df = pd.DataFrame(data=principalComponents_cifar, columns=['PC1', 'PC2'])
	principal_cifar_Df['y'] = races# df_cifar.loc[:, ['Race']].values

	plt.figure(figsize=(16, 10))
	sns.scatterplot(
		x="PC1", y="PC2",
		hue="y",
		palette=sns.color_palette("hls", 24),
		data=principal_cifar_Df,
		legend="full",
		alpha=1
	)
	plt.show()


if __name__ == '__main__':
	main()
