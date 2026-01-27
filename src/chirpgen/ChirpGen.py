import numpy as np
from scipy.signal import chirp
import torch


class ChirpGenerator:
    def __init__(self, **defaults):
        self.params = {
            "fs": 2e9,
            "f0": 300e6,
            "f1": 900e6,
            "num_samples" : 8192,
            "snr": 0,
            "method": "linear",  # linear, quadratic, logarithmic
            "no_signal": False,
        }
        self.params.update(defaults)

    def generate(self, **overrides):
        """Merges parameters from constructor
        and uses scipy chirp for IQ data
        """

        p = {**self.params, **overrides}
        t = np.linspace(0, p["num_samples"] / p["fs"], p["num_samples"], endpoint=False)
        num_samples = p["num_samples"]
        print(f"IN ChirpGenerator.generate: num_samples: {num_samples}")
        sig = chirp(
            t,
            f0=p["f0"],
            t1=p["num_samples"] / p["fs"], # Duration
            f1=p["f1"],
            method=p["method"],
            complex=True,
        )
        sig = np.column_stack((sig.real, sig.imag))
        if p["snr"] is not None:
            sig_pwr = np.sqrt((sig[:,0] ** 2) + (sig[:,1] ** 2))
      #sig_pwr = np.mean(np.abs(sig) ** 2)
            noise_pwr = sig_pwr / (10 ** (p["snr"] / 10))
            noisei = np.sqrt(noise_pwr / 2) * (np.random.normal(loc=0.0, scale=1.0, size=len(sig)))
            noiseq = np.sqrt(noise_pwr / 2) * (np.random.normal(loc=0.0, scale=1.0, size=len(sig)))
            noise = np.column_stack([noisei, noiseq])
            if p["no_signal"]:
              sig= noise
            else:
              sig = sig + noise
        print(f"Length of a signal: {len(sig)}")
        sig = sig.astype(np.float32)
        iq_tensor = torch.from_numpy(sig)

        labels = {
            "fs": p["fs"],
            "snr": p["snr"],
            "num_samples": p["num_samples"],
            "f0": p["f0"],
            "f1": p["f1"],
            "no_signal": p["no_signal"],
        }

        return iq_tensor, labels
if __name__ == "__main__":
  import matplotlib.pyplot as plt
  gen = ChirpGenerator(fs=4e9, snr=1, num_samples=8192, f0=100e6, f1=1000e6, no_signal=False)
  plot = True
  iq_tensor, label = gen.generate()

  print(f"Shape of returned sig tensor: {iq_tensor.shape}")
  if (plot):
    fs, snr, num_samples, f0, f1, no_signal = label.values()
    duration = num_samples / fs
    print(f"Labels: {label.values()}")
    print(f"SNR: {snr}")
    print(f"No signal?: {no_signal}")
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    print(f"{iq_tensor}")
    plt.style.use("dark_background")
    ax.plot(t, iq_tensor[:, 0], label="I set", color="tab:blue")
    ax.plot(t, iq_tensor[:, 1], label="Q set", color="tab:orange")
    ax.set_title("I and Q", fontsize=4)
    ax.set_xlabel("time")
    ax.set_ylabel("IQ tensor data")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 50e-9)
    plt.show()
    print(f"shape of labels: {label}")
