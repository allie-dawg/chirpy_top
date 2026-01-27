import torch
import numpy as np
from ChirpGen import ChirpGenerator


def createadataset(num_chirps=100, low_snr=10, high_snr=90):
    print(f"Generating {num_chirps} chirps with random attributes")
    fs = 4e9
    gen = ChirpGenerator(fs=fs, method="linear")
    num_samples_set = [1024, 2048, 4096, 8192, 16384]
    dataset = {"iq_tensors": [], "labels": []}

    for _ in range(num_chirps):
        cur_snr = np.random.uniform(low_snr, high_snr)
        # fs * dur = num_samples
        cur_num_samp = np.random.choice(num_samples_set)
        cur_f0 = np.random.uniform(100e6, 500e6)
        cur_f1 = np.random.uniform(1000e6, 2000e6)
        cur_has_sig = np.random.choice([True, False])
        print(f"IN datagen.py createadataset loop cur_num_samp: {cur_num_samp}") 
        x_tensor, label_dict = gen.generate(
            fs=fs,
            snr=cur_snr,
            num_samples=cur_num_samp,
            f0=cur_f0,
            f1=cur_f1,
            no_signal=cur_has_sig,
        )

        #print(f"Current labels: {label_dict}")
        dataset["iq_tensors"].append(x_tensor)
        dataset["labels"].append(label_dict)
    filename = f"dataset_of_{num_chirps}_chirps.pt"
    torch.save(dataset, filename)

    print(f"Done. Saved chirps to chirp_dataset_{num_chirps}.pt")
    return filename
