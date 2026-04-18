# ocr/test_noise.py
import tensorflow as tf
import numpy as np
from noise import add_gaussian_noise, add_salt_pepper_noise, apply_noise

# create a clean test image (28x28, all 0.5 gray)
clean = tf.ones((28, 28, 1)) * 0.5

# test 1: Gaussian noise changes pixels but stays in [0,1]
noisy_gauss = add_gaussian_noise(clean, std=0.15)
noisy_gauss_np = noisy_gauss.numpy()
assert noisy_gauss_np.min() >= 0.0
assert noisy_gauss_np.max() <= 1.0
assert not np.allclose(noisy_gauss_np, 0.5)   # must have changed
print(f"TEST 1 PASSED — Gaussian noise applied, "
      f"range=[{noisy_gauss_np.min():.3f}, {noisy_gauss_np.max():.3f}]")

# test 2: Salt-pepper noise produces some pure white and pure black pixels
noisy_sp = add_salt_pepper_noise(clean, salt_prob=0.1, pepper_prob=0.1)
noisy_sp_np = noisy_sp.numpy()
has_salt   = np.any(noisy_sp_np == 1.0)
has_pepper = np.any(noisy_sp_np == 0.0)
assert has_salt,   "Expected some salt pixels (value 1.0)"
assert has_pepper, "Expected some pepper pixels (value 0.0)"
print(f"TEST 2 PASSED — salt-pepper noise applied, "
      f"salt_count={int(np.sum(noisy_sp_np == 1.0))}, "
      f"pepper_count={int(np.sum(noisy_sp_np == 0.0))}")

# test 3: apply_noise dispatcher works
g = apply_noise(clean, noise_type="gaussian", std=0.1)
sp = apply_noise(clean, noise_type="salt_pepper", salt_prob=0.05)
none = apply_noise(clean, noise_type="none")
assert tf.reduce_all(tf.equal(none, clean))
print("TEST 3 PASSED — dispatcher works for all 3 modes")

# test 4: higher std = more visible noise
low  = add_gaussian_noise(clean, std=0.05).numpy()
high = add_gaussian_noise(clean, std=0.3).numpy()
low_variance  = np.var(low)
high_variance = np.var(high)
assert high_variance > low_variance
print(f"TEST 4 PASSED — noise intensity scales with std "
      f"(var_low={low_variance:.4f}, var_high={high_variance:.4f})")


# test 5: noise is reproducible with seed
tf.random.set_seed(42)
a = add_gaussian_noise(clean, std=0.15).numpy()
tf.random.set_seed(42)
b = add_gaussian_noise(clean, std=0.15).numpy()
assert np.array_equal(a, b), "Same seed should give same noise"
print("TEST 5 PASSED — noise is deterministic with seed")



print("\nALL NOISE TESTS PASSED")


