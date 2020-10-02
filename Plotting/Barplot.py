import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
	df = pd.read_csv("raw_with_cluster_labels.csv")
	sns.set_style("darkgrid")
	for feature in df.columns:
		figure = sns.catplot(x='Segment', y=feature, data=df, kind='box')
		path = "full df Boxplots/{}.png"
		figure.savefig(path.format(feature))
		plt.clf()


if __name__ == '__main__':
	main()
