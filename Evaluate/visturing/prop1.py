import numpy as np
import matplotlib.pyplot as plt

from jax import random, numpy as jnp
from flax.core import pop
from pnet_revisited.model import Model
from pnet_revisited.initialization import init_model
from visturing.properties import prop1 as prop
from scipy.stats import pearsonr

# Load the model
key = random.PRNGKey(42)
x = jnp.ones((1,128,128,3))
model = Model()
variables = model.init(key, x)
state, params = pop(variables, "params")
_, state = model.apply({"params": params, **state}, x, train=True, mutable=list(state.keys()))

params, state = init_model(model, params, state)

def calculate_diffs(a, b):
    a = model.apply({"params": params, **state}, a, train=False)
    b = model.apply({"params": params, **state}, b, train=False)
    return ((a-b)**2).mean(axis=(-3,-2,-1))**(1/2)

results = prop.evaluate(calculate_diffs=calculate_diffs,
                        data_path="../Data/Experiment_1/",
                        gt_path="../Data/ground_truth/")

print(f'correlations: {results["correlations"]["pearson"]}')

plt.plot(results["lambdas"], results["diffs"])
plt.show()
