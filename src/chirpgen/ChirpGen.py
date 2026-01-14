import numpy as np
from scipy.signal import chirp
import torch


class ChirpGenerator:
    def __init__(self, **defaults):
        self.params = {
            "fs": 2e9,
            "f0": 300e6,
            "f1": 900e6,
            "duration": 50e-9,
            "snr_dbFS": 0,
            "method": "linear",  # linear, quadratic, logarithmic
            "no_signal": False,
        }
        self.params.update(defaults)

    def generate(self, **overrides):
        """Merges parameters from constructor
        and uses scipy chirp for IQ data"""

        p = {**self.params, **overrides}

        t = np.linspace(0, p["duration"], int(p["fs"] * p["duration"]), endpoint=False)

        sig = chirp(
            t,
            f0=p["f0"],
            t1=p["duration"],
            f1=p["f1"],
            method=p["method"],
            complex=True,
        )

        if p["snr_dbFS"] is not None:
            sig_pwr = np.mean(np.abs(sig) ** 2)
            noise_pwr = sig_pwr / (10 ** (p["snr_dbFS"] / 10))
            noise = np.sqrt(noise_pwr / 2) * (
                np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))
            )
            if p["no_signal"]:
                sig = noise
            else:
                sig = sig + noise

        iq_tensor = torch.from_numpy(sig)

        labels = {
            "fs": p["fs"],
            "snr": p["snr_dbFS"],
            "duration": p["duration"],
            "f0": p["f0"],
            "f1": p["f1"],
            "no_signal": p["no_signal"],
        }

        return iq_tensor, labels
