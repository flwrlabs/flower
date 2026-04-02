import argparse
import random

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from whisper_example.dataset import get_encoding_fn, prepare_silences_dataset
from whisper_example.model import (
    construct_balanced_sampler,
    eval_model,
    get_model,
    train_one_epoch,
)

random.seed(1989)
torch.set_float32_matmul_precision(
    "high"
)  #  If “high” or “medium” are set then the TensorFloat32 is used
NUM_CLASSES = 12
REMOVE_COLS = ["file", "audio", "label", "is_unknown", "speaker_id", "utterance_id"]
parser = argparse.ArgumentParser(description="Whisper centralised")

parser.add_argument("--checkpoint", type=str, help="path to classifier`s checkpoint")
parser.add_argument(
    "--epochs", type=int, default=3, help="Number of epochs of training."
)
parser.add_argument(
    "--compile", action="store_true", help="compiles model (pytorch 2.0+ only)"
)


def save_classifier(classifier, acc: float):
    filename = f"classifier_{acc:.4f}.pt"
    torch.save(classifier.cpu().state_dict(), filename)
    return filename


def main():
    args = parser.parse_args()

    # load train and test partitions
    sc = load_dataset("speech_commands", "v0.02", split="train", token=False)
    sc_val = load_dataset("speech_commands", "v0.02", split="validation", token=False)
    sc_test = load_dataset("speech_commands", "v0.02", split="test", token=False)

    # pre-process dataset
    # ! If you know how to speedup this pre-processing stage, please do let us know!
    # ! Become a contributor by proposing as a new PR !
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    prepare_dataset_fn = get_encoding_fn(processor)
    og_threads = torch.get_num_threads()
    print(f"{og_threads = }")
    torch.set_num_threads(
        1
    )  # not clear to me why we need this in order to be able to use `num_proc > 1 for .map`
    train_encoded = sc.map(prepare_dataset_fn, num_proc=4, remove_columns=REMOVE_COLS)
    val_encoded = sc_val.map(prepare_dataset_fn, num_proc=4, remove_columns=REMOVE_COLS)
    test_encoded = sc_test.map(
        prepare_dataset_fn, num_proc=4, remove_columns=REMOVE_COLS
    )

    # create and pre-process the dataset of silences
    silences_dataset = prepare_silences_dataset(sc, ratio_silence=0.1)
    # ! You might want to save this encoded_silences dataset to disk, so this stage is not
    # ! needed each time you run the code. Alternatively, this silence generation could be
    # ! implemented as part of a `collate_fn` in the standard PyTorch dataloader...
    encoded_silences = silences_dataset.map(
        prepare_dataset_fn, num_proc=4, remove_columns=REMOVE_COLS
    )
    full_train_dataset = concatenate_datasets([train_encoded, encoded_silences])

    torch.set_num_threads(og_threads)

    lbls = set(full_train_dataset["targets"])
    print(f"{lbls = }")
    # Construct a balanced sampler so batches roughly contain the same number
    # of samples from each class
    sampler = construct_balanced_sampler(full_train_dataset)

    # Prepare dataloaders
    train_dataset = full_train_dataset.with_format("torch", columns=["data", "targets"])
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=False, num_workers=4, sampler=sampler
    )
    val_encoded = val_encoded.with_format("torch", columns=["data", "targets"])
    val_loader = DataLoader(val_encoded, batch_size=64, num_workers=4)
    test_dataset = test_encoded.with_format("torch", columns=["data", "targets"])
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)

    # Model to cuda, set criterion, classification layer to train and optimiser
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder, classifier = get_model(device, num_classes=12)
    criterion = torch.nn.CrossEntropyLoss()

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint = }")
        classifier.load_state_dict(torch.load(args.checkpoint))
    classifier = classifier.to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001)
    encoder.eval()

    # Let's count the size of the classification head
    classifier_head_params = sum(p.numel() for p in classifier.parameters())
    print(f"{classifier_head_params = }")

    # Eval initial model
    loss, accuracy = eval_model(encoder, classifier, criterion, val_loader, device)
    print(f"Initial (loss, acc): {loss = }, {accuracy = }")
    best = [-float("inf"), None]
    for e in range(args.epochs):
        print(f"Epoch: {e}")
        train_one_epoch(encoder, classifier, optimizer, criterion, train_loader, device)
        loss, accuracy = eval_model(encoder, classifier, criterion, val_loader, device)
        last_saved = save_classifier(classifier, accuracy)
        if accuracy > best[0]:
            best[0] = accuracy
            best[1] = last_saved
        print(f"VALIDATION ---> {loss = }, {accuracy = }")

    print("Training done...")
    print("Evaluating test set. Loading best model")
    classifier.load_state_dict(torch.load(best[1]))
    loss, accuracy = eval_model(encoder, classifier, criterion, test_loader, device)
    print(f"TEST ---> {loss = }, {accuracy = }")


if __name__ == "__main__":
    main()
