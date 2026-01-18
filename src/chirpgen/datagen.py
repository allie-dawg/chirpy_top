import torch
import numpy as np
from ChirpGen import ChirpGenerator


def datagen(num_chirps=100, low_snr=10, high_snr=90):
    print(f"Generating {num_chirps} chirps with random attributes")
    gen = ChirpGenerator(fs=4e9, method="linear")

    dataset = {"iq_tensors": [], "labels": []}

    for _ in range(num_chirps):
        cur_snr = np.random.uniform(low_snr, high_snr)
        cur_dur = np.random.choice([100e-9, 200e-9, 300e-9, 400e-9])
        cur_f0 = np.random.uniform(100e6, 500e6)
        cur_f1 = np.random.uniform(1000e6, 2000e6)
        cur_has_sig = np.random.choice([True, False])
        x_tensor, label_dict = gen.generate(
            snr_dbFS=cur_snr,
            duration=cur_dur,
            f0=cur_f0,
            f1=cur_f1,
            no_signal=cur_has_sig,
        )

        dataset["iq_tensors"].append(x_tensor)
        dataset["labels"].append(label_dict)
    filename = f"dataset_of_{num_chirps}_chirps.pt"
    torch.save(dataset, filename)

    print(f"Done. Saved chirps to chirp_dataset_{num_chirps}.pt")
    return filename
