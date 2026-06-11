import numpy as np
import matplotlib.pyplot as plt

from visturing.properties.noise import generate_noise, generate_noise_iters, generate_plain
from visturing.properties import prop3_4 as prop

from paramperceptnet.pretrained import load_param_pretrained

# Load pretrained model
model, variables = load_param_pretrained()
params = variables["params"]
state = variables["state"]
inputs = np.ones(shape=(1,128,128,3))
pred, state = model.apply({"params": params, **state}, inputs, train=True, mutable=list(state.keys()))

def calculate_diffs(a, b):
    a = model.apply({"params": params, **state}, a, train=False)
    b = model.apply({"params": params, **state}, b, train=False)
    return ((a-b)**2).mean(axis=(-3,-2,-1))**(1/2)

res = prop.evaluate(calculate_diffs=calculate_diffs,
              data_path="../Data/Experiment_3_4/",
              gt_path="../Data/ground_truth/")

for c in res["diffs_s"]:
    plt.plot(c)
plt.show()
print(res["correlations"])
