import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
	df = pd.read_csv("Grouped_by_race_with_cluster_labels.csv")
	sns.set_style("darkgrid")
	for feature in df.columns:
		figure = sns.catplot(x='Segment', y=feature, data=df, kind='box', showfliers=False)
		path = "Boxplots/{}.png"
		figure.savefig(path.format(feature))
		plt.clf()


def plot_races():
	df = pd.read_csv("raw_with_cluster_labels_3_clusters.csv")
	df = df.sort_values("Race")
	sns.set_style("darkgrid")
	df = df[df['Race'] == 'Halfling']
	total = float(len(df))
	print(total)
	figure = sns.countplot(x='Segment', data=df)
	plt.ylabel("Number of Goblins")
	for p in figure.patches:
		height = p.get_height()
		figure.text(p.get_x() + p.get_width() / 2.,
		            height + 3,
		            '{:1.2f}%'.format((height / total)*100),
		            ha="center")
	path = "full df 3 clusters/Race to segment count.png"
	# figure.savefig(path)
	plt.show()


def plot_feat_per_race():
	df = pd.read_csv("raw_with_cluster_labels_3_clusters.csv")
	sns.set_style("darkgrid")
	df = df.sort_values("Race")
	# df = df[(df['Race'] == 'Orc') | (df['Race'] == 'Vampire')]
	total = float(len(df))
	print(total)
	figure = sns.catplot(x='Breaks', y='Race', orient='horizontal',  data=df, kind='box', showfliers=False)
	path = "full df 3 clusters/Race to segment count.png"
	# figure.savefig(path)
	plt.show()


if __name__ == '__main__':
	# plot_races()
	# main()
	plot_feat_per_race()

