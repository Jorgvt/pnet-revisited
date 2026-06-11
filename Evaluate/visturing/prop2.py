import numpy as np
import matplotlib.pyplot as plt

from jax import random, numpy as jnp
from flax.core import pop
from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from visturing.properties import prop2 as prop
from scipy.stats import pearsonr

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
square_size = (64, 64)


def calculate_diffs(a, b):
    a = model.apply({"params": params, **state}, a, train=False)
    b = model.apply({"params": params, **state}, b, train=False)
    return ((a-b)**2).mean(axis=(-3,-2,-1))**(1/2)

res = prop.evaluate_gen(calculate_diffs=calculate_diffs,
                        img_size=img_size,
                        square_size=square_size,
                        return_stimuli=True,
                        return_gt=True)
results, stimuli, correlations, gt = res.results, res.stimuli, res.correlations, res.gt

print(f"correlations: {correlations}")

fig, axes = plt.subplots(1,3)
for (name, res), ax in zip(results.items(), axes.ravel()):
    for d in res:
        ax.plot(d)
    ax.set_title(name)
plt.show()


def plots(data):
    fig, axes = plt.subplots(*data.shape[:2])
    for a, axs in zip(data, axes):
        for aa, ax in zip(a, axs):
            ax.imshow(aa)
            ax.axis("off")
    plt.show()

for name, stim in stimuli.items():
    plots(stim)

fig, axes = plt.subplots(1,3)
for (name, res), ax in zip(gt.items(), axes.ravel()):
    for d in res:
        ax.plot(d)
    ax.set_title(name)
plt.show()

fig, axes = plt.subplots(1,1)
for (name, g), (name, res), in zip(gt.items(), results.items()):

    a = np.array([g.ravel()]).ravel()
    b = np.array([res.ravel()]).ravel()
    print(f"{name}: Pearson -> {pearsonr(a,b)[0]}")
    axes.scatter(a, b)
    axes.set_title(name)
plt.show()
