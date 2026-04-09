import argparse
import os
import pickle
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

def check_folders(directory, required_folders):
    for folder in required_folders:
        folder_path = os.path.join(directory, folder)
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"Error: The folder '{folder}' is missing in the input directory '{directory}'.")
            exit(1)

def get_accuracies(path):
    with open(path, 'rb') as f:
            hist = pickle.load(f)['history']

    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    rounds, values = zip(*metric_dict["accuracy"])
    return np.asarray(rounds), np.asarray(values) * 100


def get_round_when_x_acc(accuracies, rounds, x):
    for i, acc in enumerate(accuracies):
        if acc >= x:
            return rounds[i]
    return rounds[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_type", choices=["table", "figure"])
    parser.add_argument("--output_path")

    if parser.parse_known_args()[0].output_type == "table":
        parser.add_argument("--input_directory")
    elif parser.parse_known_args()[0].output_type == "figure":
        parser.add_argument("--dataset_name")
        parser.add_argument("--input_directory")

    args = parser.parse_args()

    if args.output_type == "table":
        input_directory = args.input_directory
        
        required_folders = [
            "femnist_0", "femnist_1", "femnist_2",
            "cifar10_0", "cifar10_1", "cifar10_2",
            "cifar100_0", "cifar100_1", "cifar100_2"
        ]

        check_folders(input_directory, required_folders)

        max_femnist = [f'{max(accs):.2f}' for rnds, accs in [get_accuracies(f'{input_directory}/femnist_{i}/results.pkl') for i in [0, 1, 2]]]
        max_cifar10 = [f'{max(accs):.2f}' for rnds, accs in [get_accuracies(f'{input_directory}/cifar10_{i}/results.pkl') for i in [0, 1, 2]]]
        max_cifar100 = [f'{max(accs):.2f}' for rnds, accs in [get_accuracies(f'{input_directory}/cifar100_{i}/results.pkl') for i in [0, 1, 2]]]

        rnd_femnist = [f'{get_round_when_x_acc(accs, rnds, 80):.0f}' for rnds, accs in [get_accuracies(f'{input_directory}/femnist_{i}/results.pkl') for i in [0, 1, 2]]]
        rnd_cifar10 = [f'{get_round_when_x_acc(accs, rnds, 70):.0f}' for rnds, accs in [get_accuracies(f'{input_directory}/cifar10_{i}/results.pkl') for i in [0, 1, 2]]]
        rnd_cifar100 = [f'{get_round_when_x_acc(accs, rnds, 40):.0f}' for rnds, accs in [get_accuracies(f'{input_directory}/cifar100_{i}/results.pkl') for i in [0, 1, 2]]]

        fig, ax = plt.subplots()
        table = ax.table(
            cellText=[list(i) for i in zip(*[max_femnist, rnd_femnist, max_cifar10, rnd_cifar10, max_cifar100, rnd_cifar100])],
            cellText=[list(i) for i in zip(*[max_cifar10, rnd_cifar10, max_cifar100, rnd_cifar100])],
            cellLoc="center",
            colLabels=['FEMNIST (max test, 200 rnds)', 'FEMNIST (rnd to 80%)', 'CIFAR10 (max test, 500 rnds)', 'CIFAR10 (rnd to 70%)', 'CIFAR100 (max test, 500 rnds)', 'CIFAR100 (rnd to 40%)'],
            colLabels=['CIFAR10\n(500 rnds)', 'CIFAR10\n(rnd to 70%)', 'CIFAR100\n(500 rnds)', 'CIFAR100\n(rnd to 40%)'],
            rowLabels=["FedAvg", "NaiveMix", "FedMix"],
            loc="center"
        )

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_height(.1)

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        ax.axis("off")
        plt.savefig(args.output_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    elif args.output_type == "figure":
        dataset_name = args.dataset_name
        input_directory = args.input_directory
    
        required_folders = [f"{dataset_name}_0", f"{dataset_name}_1", f"{dataset_name}_2"]
    
        check_folders(input_directory, required_folders)

        all_rounds, all_accuracies, max_accuracies = {}, {}, {}
        for i, lbl in enumerate(["FedAvg", "NaiveMix", "FedMix"]):
            all_rounds[lbl], all_accuracies[lbl] = get_accuracies(f'{input_directory}/{dataset_name}_{i}/results.pkl')
            max_accuracies[lbl] = max(all_accuracies[lbl])

        def average_list_values(values, n=10):
            values = [values[0], ] * n + list(values) + [values[-1], ] * n
            averaged_values = []
            for i, v in enumerate(values[n:-n]):
                averaged_values.append(mean((values[i:i+2*n])))

            return averaged_values

        for (lbl, accuracies), (_, rounds) in zip(all_accuracies.items(), all_rounds.items()):
            plt.plot(rounds, average_list_values(accuracies, 5), label=lbl, alpha=0.6)

        plt.xlabel('Comm. Round')
        plt.ylabel('Test Acc. (%)')

        if dataset_name == 'femnist':
            lim1 = 70
            lim2 = 85
        elif dataset_name == 'cifar10':
            lim1 = 55
            lim2 = 85
        elif dataset_name == 'cifar100':
            lim1 = 45
            lim2 = 60

        plt.ylim(lim1, lim2)

        plt.grid(True, 'both', 'both')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.75*len(rounds), lim1+2, 'Max Accuracy:\n' + '\n'.join([f'{l}: {a:.2f}' for l, a in max_accuracies.items()]), bbox=props)
        
        plt.legend()

        plt.savefig(args.output_path)
        plt.close()

if __name__ == "__main__":
    main()
