import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
	df = pd.read_csv("Grouped_by_race_with_cluster_labels.csv")
	sns.set_style("darkgrid")
	sns.catplot(x='Segment', y='Touchdowns', ci=None, data=df)
	plt.show()


if __name__ == '__main__':
	main()
