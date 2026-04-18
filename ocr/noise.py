# ocr/noise.py
"""
Two noise profiles for training and evaluation:
  1. Gaussian noise — additive random noise per pixel
  2. Salt-and-pepper noise — random pixels set to min or max intensity

Both preserve the underlying character structure while simulating
realistic image corruption (scan artifacts, sensor noise, poor lighting).
"""
import tensorflow as tf


def add_gaussian_noise(image: tf.Tensor, std: float = 0.15) -> tf.Tensor:
    """
    Add Gaussian noise to an image.
    
    Args:
        image: tensor with values in [0, 1]
        std:   standard deviation of the noise (higher = more noise)
    
    Returns:
        noisy image, clipped to [0, 1]
    """
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=std)
    noisy = image + noise
    return tf.clip_by_value(noisy, 0.0, 1.0)


def add_salt_pepper_noise(
    image: tf.Tensor,
    salt_prob: float = 0.02,
    pepper_prob: float = 0.02
) -> tf.Tensor:
    """
    Apply salt-and-pepper noise to an image.
    
    Salt = random white pixels (max intensity)
    Pepper = random black pixels (min intensity)
    
    Args:
        image:       tensor with values in [0, 1]
        salt_prob:   probability of each pixel becoming white
        pepper_prob: probability of each pixel becoming black
    """
    shape  = tf.shape(image)
    rand   = tf.random.uniform(shape, minval=0.0, maxval=1.0)

    # salt: set to 1 where rand < salt_prob
    salted   = tf.where(rand < salt_prob, tf.ones_like(image), image)

    # pepper: set to 0 where rand is in next band
    peppered = tf.where(
        (rand >= salt_prob) & (rand < salt_prob + pepper_prob),
        tf.zeros_like(salted),
        salted
    )

    return peppered


def apply_noise(
    image: tf.Tensor,
    noise_type: str = "gaussian",
    **kwargs
) -> tf.Tensor:
    """
    Unified noise application function for training pipelines.
    
    Args:
        image:      input tensor
        noise_type: one of "gaussian", "salt_pepper", "none"
        kwargs:     parameters passed to specific noise function
    """
    if noise_type == "gaussian":
        return add_gaussian_noise(image, std=kwargs.get("std", 0.15))
    elif noise_type == "salt_pepper":
        return add_salt_pepper_noise(
            image,
            salt_prob=kwargs.get("salt_prob", 0.02),
            pepper_prob=kwargs.get("pepper_prob", 0.02)
        )
    elif noise_type == "none":
        return image
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")