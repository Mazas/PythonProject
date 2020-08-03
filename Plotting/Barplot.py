import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
	df = pd.read_csv("raw_cluster_labels.csv")
	sns.set_style("darkgrid")
	sns.catplot(x='Casualties', y='Segment', data=df, kind='box', orient='h')
	plt.show()


if __name__ == '__main__':
	main()
