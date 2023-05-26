import json

def main():

    file_path = "metrics/metrics.json"

    with open(file_path, 'r') as file:
        json_data = file.read()
    data = json.loads(json_data)

    vanilla_avg_accuracy, directed_avg_accuracy = 0, 0
    for approach in data.keys():
        for key in data[approach].keys():
            if approach == 'vanilla':
                vanilla_avg_accuracy += data[approach][key]['test_accuracies'][-1]
            if approach == 'directed':
                directed_avg_accuracy += data[approach][key]['test_accuracies'][-1]

    vanilla_avg_accuracy /= 5
    directed_avg_accuracy /= 5
    print('vanilla_avg_accuracy', vanilla_avg_accuracy)
    print('directed_avg_accuracy', directed_avg_accuracy)

if __name__ == "__main__":
    main()