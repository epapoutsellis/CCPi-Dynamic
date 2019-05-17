import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Paste your code here
fig, ax = plt.subplots()
ims = []
N = 500
x = np.random.rand(N, 50, 2)
fakenp = np.random.rand(N, 50, 2)
for i in range(N):
    im1, = plt.plot(x[i, :, 0], x[i, :, 1], 'b.')
    im2, = plt.plot(fakenp[i, :, 0], fakenp[i, :, 1], 'rx')
    ims.append([im1, im2])
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                repeat_delay=1000)
ani.save('sample.mp4', writer='ffmpeg')