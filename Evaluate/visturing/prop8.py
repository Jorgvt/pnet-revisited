import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

from jax import random, numpy as jnp
from flax.core import pop
from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from visturing.properties import prop8 as prop

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
Cs_mask = np.linspace(0, 0.2, num=5)[1:]
Cs_mask = (((img_size[0]/fs)**2)/((2*R0)**2))**(1/2)*Cs_mask

c = 1
n_iters = 2

freqs = np.arange(1, 17, step=1)
freqs_mask = np.array([2, 4, 6, 8, 10])

# freqs = np.array([4.])
# freqs_mask = np.array([3.])

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
                )
results, freqs, stimuli, correlation = res.results, res.freqs, res.stimuli, res.correlations

print(f"Correlation: {correlation}")
