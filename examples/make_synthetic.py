"""Write a synthetic Fe/Cu/Zn/Pb XRF spectrum (Si detector) to examples/synthetic.txt"""
import numpy as np
from xrfit import core
rng = np.random.default_rng(1)
cfg = core.Config(elements=["Fe", "Cu", "Zn", "Pb-L"], zero=0.02, gain=0.01, noise=0.12,
                  tail_frac=0.05, esc_frac=0.01)
ch = np.arange(2048.0)
p = cfg.params()
A, names, _ = core.design(cfg, p, ch)
amp = np.array([40000, 15000, 8000, 5000.0])
e = p["zero"] + p["gain"] * ch
bg = 150 * np.exp(-e / 6) + 5
y = rng.poisson(A @ amp + bg)
np.savetxt("examples/synthetic.txt", y, fmt="%d")
print("true areas:", dict(zip(names, amp)))
