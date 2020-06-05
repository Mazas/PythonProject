# Python program to convert
# JSON file to CSV


import json
import csv
import glob


def main():
	# headers to the CSV file
	# Counter variable used for writing
	empty_logs = open('empty_logs.txt', 'w')
	empty = 0
	count = 0
	# open a file for writing
	data_file = open('data_file.csv', 'w', encoding='utf-8', newline='')

	# create the csv writer object
	csv_writer = csv.writer(data_file)

	# Opening JSON file and loading the data
	# into the variable data
	for file in glob.glob('C:/Users/ivana/Desktop/Uni/Mailed Resources/www.mordrek.com/goblinSpy/Replays/*/*.json'):
		with open(file, encoding='utf-8') as json_file:
			data = json.load(json_file)

			coach_data = data['TeamStats']

		for coach in coach_data:
			if coach['Rolls'] is None:
				print("Empty Log File: ", file)
				empty_logs.write('%s \n' % file)
				empty += 1
				continue
			# delete unused features
			del [coach['CoachName']]
			del [coach['TeamName']]
			del [coach['LeagueName']]
			del [coach['CompetitionName']]
			del [coach['PlayerStats']]
			del [coach['Games']]
			del [coach['Rolls']]
			del [coach['CoachId']]
			del [coach['TeamId']]
			del [coach['LeagueId']]
			del [coach['GamesPlayed']]
			del [coach['RealGamesPlayed']]

			if 'Mercenary' in coach['Race']:
				continue

			if count == 0:
				coach.update({"GameWon": False})
				# Writing headers of CSV file
				header = coach.keys()
				csv_writer.writerow(header)
				count += 1

			# Writing data of CSV file
			coach.update({"GameWon": coach['Score'] > coach['ScoreAgainst']})
			csv_writer.writerow(coach.values())
			json_file.close()
	print("%d empty log files." % empty)
	data_file.close()
	empty_logs.close()
	print("Finished compiling data set.")


if __name__ == "__main__":
	main()
