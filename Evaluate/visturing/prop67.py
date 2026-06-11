import numpy as np
import matplotlib.pyplot as plt

from jax import random, numpy as jnp
from flax.core import pop
from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from visturing.properties import prop6_7 as prop

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
Cs = np.linspace(0.01, 0.2, num=10)
c = 1
n_iters = 2
freqs = np.arange(1, 17, step=1)
print(f"Freqs: {freqs}")

def calculate_diffs(a, b):
    a = model.apply({"params": params, **state}, a, train=False)
    b = model.apply({"params": params, **state}, b, train=False)
    return ((a-b)**2).mean(axis=(-3,-2,-1))**(1/2)

res = prop.evaluate_gen(
                calculate_diffs=calculate_diffs,
                img_size=img_size,
                freqs=freqs,
                L=L,
                Cs=Cs,
                fs=fs,
                sigma_mask=sigma_mask,
                n_iters=n_iters,
                theta=theta,
                delta_theta=delta_theta,
                return_stimuli=True,
                )
results, freqs, stimuli, correlation = res.results, res.freqs, res.stimuli, res.correlations

print(f"Correlation: {correlation}")


for stimuli_ in stimuli.values():
    fig, axes = plt.subplots(stimuli["achrom"].shape[1], stimuli["achrom"].shape[2])
    for i, axs in enumerate(axes):
        for j, ax in enumerate(axs):
            ax.imshow(stimuli_[0,i,j])
            ax.axis("off")
    plt.show()

fig, axes = plt.subplots(1,3)

for i, (k, v) in enumerate(results.items()):
    for j, vv in enumerate(v):
        axes[i].plot(vv, label=Cs[j])
    axes[i].set_title(k)
plt.legend()
plt.show()
