import numpy as np
import matplotlib.pyplot as plt

from jax import random, numpy as jnp
from flax.core import pop
from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from visturing.properties import prop5 as prop

# Load the model
key = random.PRNGKey(42)
x = jnp.ones((1,128,128,3))
model = Model()
variables = model.init(key, x)
state, params = pop(variables, "params")
_, state = model.apply({"params": params, **state}, x, train=True, mutable=list(state.keys()))

params, state = init_model(model, params, state)

# Config
img_size = (128, 128)
fs = 128
fm = 0.25
fM = 30
theta = 0
delta_theta = 30
sigma_mask = 0.3
R0 = 0.15
N = 15
L = 40
C = 0.025
C_mask = 0.0125
C_mask = (((img_size[0]/fs)**2)/((2*R0)**2))**(1/2)*C_mask
n_iters = 1
freqs = np.arange(1, 17, step=1)
freqs_mask = np.array([2, 4, 6, 8, 10])

def calculate_diffs(a, b):
    a = model.apply({"params": params, **state}, a, train=False)
    b = model.apply({"params": params, **state}, b, train=False)
    return ((a-b)**2).mean(axis=(-3,-2,-1))**(1/2)

res = prop.evaluate_gen(calculate_diffs,
                  img_size=img_size,
                  freqs=freqs,
                  freqs_mask=freqs_mask,
                  L=L,
                  C=C,
                  C_mask=C_mask,
                  fs=fs,
                  sigma_mask=sigma_mask,
                  n_iters=n_iters,
                  theta=theta,
                  delta_theta=delta_theta,
                  return_stimuli=True)
diffs, freqs, stimuli, correlation = res.results, res.freqs, res.stimuli, res.correlations

for k, v in stimuli.items():
    print(f"{k}: {v.min()}, {v.max()}")
print(correlation)

# fig, axes = plt.subplots(3, stimuli["achrom"].shape[1])
# for im, ax in zip(stimuli["achrom"][0], axes[0].ravel()):
#     ax.imshow(im, vmin=0, vmax=1)
#     ax.axis("off")
# for im, ax in zip(stimuli["red-green"][0], axes[1].ravel()):
#     ax.imshow(im, vmin=0, vmax=1)
#     ax.axis("off")
# for im, ax in zip(stimuli["yellow-blue"][0], axes[2].ravel()):
#     ax.imshow(im, vmin=0, vmax=1)
#     ax.axis("off")
# plt.show()

fig, axes = plt.subplots(1,3)
for (k, v), ax in zip(diffs.items(), axes.ravel()):
    for i, vv in enumerate(v):
        ax.plot(freqs, vv, label=freqs_mask[i])
    ax.set_title(k)
plt.legend()
plt.show()
# fig, axes = plt.subplots(1,2)
# for k, v in diffs.items():
#     axes[0].plot(freqs, v, label=k)
# # for k, v in gt.items():
# #     axes[1].plot(freqs, v, label=k)
# plt.legend()
# plt.show()
