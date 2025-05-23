import matplotlib.pyplot as plt
import numpy as np

a = np.fromfile("output.bin", dtype=np.uint32)

N = int(np.sqrt(a.shape[0] // 6))
assert N * N * 6 == a.shape[0], \
    f"Read {a.shape[0]} values, this has to be divisible by (3 * N * 2 * N)"
a = a.reshape(3 * N, 2 * N).transpose()
print(f"Matrix has shape {a.shape}")
print(f"Minimal value = {np.amin(a)} Maximal value = {np.amax(a)} Sum = {np.sum(a)}")
plt.clf()
plt.pcolormesh(a, cmap='plasma')
dpi = 100
plt.gcf().set_size_inches(a.shape[1] / dpi, a.shape[0] / dpi)
plt.savefig('output.png')
print("Output written to 'output.png'")
