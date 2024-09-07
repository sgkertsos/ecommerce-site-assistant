import json
import csv

def convert_json_to_csv():

    # Open json file
    with open('./data/data.json', 'r') as json_file:
        data = json.load(json_file)

    # Create a csv file
    with open('./data/data2.csv', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["question", "answer"])

        writer.writeheader()

        for entry in data["questions"]:
            writer.writerow(entry)
