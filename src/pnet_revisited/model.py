from typing import Any
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange
from fxlayers.layers import pad_same_from_kernel_size, GDNGamma
from perceptualtests.color_matrices import Mng2xyz, Mxyz2atd

# Import custom layers from local layers module
from .layers import CenterSurroundLogSigmaK, DN, GaborLayerGammaHumanLike_, GDNControl

def ng2atd(img):
    return img @ Mng2xyz.T @ Mxyz2atd.T


class Model(nn.Module):

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs):

        outputs = GDNGamma()(inputs)
        outputs = ng2atd(outputs)

        outputs = pad_same_from_kernel_size(outputs, kernel_size=63, mode="symmetric")
        outputs = CenterSurroundLogSigmaK(
                            # features=3, kernel_size=31, fs=32,
                            # xmean=(31/32)/2,
                            # ymean=(31/32)/2,
                            features=3, kernel_size=64, fs=128,
                            xmean=(63/128)/2,
                            ymean=(63/128)/2,
                            normalize_prob=False,
                            normalize_energy=False,
                            normalize_sum=True,
                            substraction_factor=0.95,
                            padding="VALID")(outputs, **kwargs)

        outputs = DN(kernel_size=31, fs=31, apply_independently=True, normalize_energy=False, normalize_prob=False, normalize_sum=True)(outputs, **kwargs)

        outputs = nn.max_pool(outputs, window_shape=(4,4), strides=(4,4))

        outputs = pad_same_from_kernel_size(outputs, kernel_size=31, mode="symmetric")
        outputs, fmean, theta_mean = GaborLayerGammaHumanLike_(
            n_scales=[4, 2, 2],
            n_orientations=[8, 8, 8],
            kernel_size=31,
            fs=32,
            xmean=32 / 32 / 2,
            ymean=32 / 32 / 2,
            strides=1,
            padding="VALID",
            normalize_prob=False,
            normalize_energy=True,
            zero_mean=True,
            use_bias=False,
            train_A=False,
        )(outputs, return_freq=True, return_theta=True, **kwargs)

        outputs = GDNControl(kernel_size=31, fs=32, normalize_prob=False, normalize_energy=False, normalize_sum=True)(outputs, fmean, theta_mean, **kwargs)

        return outputs

class ModelCls(nn.Module):
    config: Any

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs):
        outputs = Model(name="perceptnet")(inputs, **kwargs)
        # outputs = reduce(outputs, "b h w c -> b c", reduction="mean")
        if self.config.GAP:
            outputs = outputs.mean(axis=(1,2))
        else:
            outputs = rearrange(outputs, "b h w c -> b (h w c)")
        outputs = nn.Dense(features=10)(outputs)
        return outputs

class SimpleDecoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        # First upsampling by 2: (b, h/4, w/4, 130) -> (b, h/2, w/2, 64)
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        # Second upsampling by 2: (b, h/2, w/2, 64) -> (b, h, w, 32)
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        # Projection back to RGB: (b, h, w, 32) -> (b, h, w, 3)
        x = nn.Conv(features=3, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.sigmoid(x)
        return x

class ModelDenoising(nn.Module):
    @nn.compact
    def __call__(self, inputs, **kwargs):
        # Encoder: extract perceptual representations
        features = Model(name="perceptnet")(inputs, **kwargs)
        
        # Determine target dimensions for the feature maps before 4x upsampling
        h, w = inputs.shape[1], inputs.shape[2]
        h_target = (h + 3) // 4
        w_target = (w + 3) // 4
        
        # Dynamically pad features to match the target dimensions
        pad_h = h_target - features.shape[1]
        pad_w = w_target - features.shape[2]
        features = jnp.pad(features, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
        
        # Decoder: reconstruct original image
        reconstruction = SimpleDecoder(name="decoder")(features)
        
        # Crop the output back to the original input resolution
        return reconstruction[:, :h, :w, :]
