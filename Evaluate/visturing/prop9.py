
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

from jax import random, numpy as jnp
from flax.core import pop
from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from visturing.properties import prop9 as prop

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

Cs = np.array([0.025])
Cs_mask = np.array([0.0125])
Cs_mask = (((img_size[0]/fs)**2)/((2*R0)**2))**(1/2)*Cs_mask

c = 1
n_iters = 2

freqs_mask = np.arange(1, 17, step=1)
freqs = np.array([2, 4, 6, 8, 10])

theta_mask = np.array([0.])

def calculate_diffs(a, b):
    a = model.apply({"params": params, **state}, a, train=False)
    b = model.apply({"params": params, **state}, b, train=False)
    return ((a-b)**2).mean(axis=(-3,-2,-1))**(1/2)

res = prop.evaluate_gen(
                calculate_diffs=calculate_diffs,
                img_size=img_size,
                freqs=freqs,
                freqs_mask=freqs_mask,
                L=L,
                Cs=Cs,
                Cs_mask=Cs_mask,
                fs=fs,
                sigma_mask=sigma_mask,
                n_iters=n_iters,
                theta=theta,
                theta_mask=theta_mask,
                delta_theta=delta_theta,
                return_stimuli=True,
                return_gt=True,
                )
results, freqs, stimuli, correlation, gt = res.results, res.freqs, res.stimuli, res.correlations, res.gt

print(f"Correlation: {correlation}")

# fig, axes = plt.subplots(stimuli["achrom"].shape[1], stimuli["achrom"].shape[2])

# for stimuli_ in stimuli.values():
#     for i, axs in enumerate(axes):
#         for j, ax in enumerate(axs):
#             ax.imshow(stimuli_[0,i,j])
#             ax.axis("off")
#     plt.show()

fig, axes = plt.subplots(1,3)
for ax, (k, v) in zip(axes.ravel(), results.items()):
    print(f"{k}: {v.shape}")
    v = v.squeeze()
    for i, a in enumerate(v.T):
        ax.plot(a, label=freqs[i])
plt.legend()
plt.show()

fig, axes = plt.subplots(1,3)
for ax, (k, v) in zip(axes.ravel(), gt.items()):
    print(f"{k}: {v.shape}")
    v = v.squeeze()
    for i, a in enumerate(v.T):
        ax.plot(a, label=freqs[i])
plt.legend()
plt.show()


# fig, axes = plt.subplots(1,3)

# for i, (k, v) in enumerate(results.items()):
#     for j, vv in enumerate(v):
#         axes[i].plot(vv, label=freqs[j])
#     axes[i].set_title(k)
# plt.legend()
# plt.show()
