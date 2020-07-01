import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

import os
from os import path


def main():
	raw_data = pd.read_csv("data_file.csv")
	grouped_by_race = raw_data.groupby("Race").count().reset_index()

	# encoding labels
	label_encoder = LabelEncoder()
	label_encoder.fit(grouped_by_race["Race"])
	grouped_by_race["Race"] = label_encoder.transform(grouped_by_race["Race"])
	# print(grouped_by_race.dtypes)

	races = label_encoder.inverse_transform(grouped_by_race["Race"])

	# DENDOGRAM
	# plt.figure(figsize=(10, 7))
	# plt.title("Race Dendograms")
	# dend = shc.dendrogram(shc.linkage(grouped_by_race, method='ward'), orientation="right", labels=races)
	# plt.show()

	print(os.getcwd())

	columns = raw_data.columns
	for col in columns:
		if col == "Race":
			continue

		bplot = sns.catplot(y=col, x='Race', data=raw_data, kind="bar", height=10, palette="muted", aspect=3)
		# bplot.set_xticklabels(rotation=45, labels=races)
		file_path = "{0}/CountGraph/{1}.png"
		print(col)
		plt.savefig(path.abspath(file_path.format(os.getcwd(), col)))
		plt.close()


# BOXPLOT
# bplot = sns.boxplot(y='Value', x='Race', data=raw_data, width=0.5, palette="muted")
# plt.show()


if __name__ == "__main__":
	main()
