import math
import matplotlib.pyplot as plt
import numpy as np
# predict:
# pre_x_k = A * pose_x_k-1 + B * u_k-1
# pre_P_k = A * pose_P_k-1 * transpose_A + Q

# update:
# K_k = (pre_P_k * transpose_H) / (H * pre_P_k * transpose_H + R)
# post_x_k = pre_x_k + K_k * (z_k - H * pre_x_k)
# post_P_k = (I - K_k * H) * pre_P_k

if __name__ == "__main__":
  N = 300
  CON = 25
  x = np.zeros((1, N))
  y = math.sqrt(2) * np.random.randn(1,N) + CON

  x[0, 0] = 1
  p = 10

  Q = np.cov(np.random.rand(1,N))
  R = np.cov(np.random.rand(1,N))

  for k in range(1, N):
      x[0, k] = x[0, k - 1]
      p = p + Q

      k_g = p / (p + R)
      x[0, k] = x[0, k] + k_g * (y[0, k] - x[0, k])
      p = (1 - k_g) * p

  x_label = np.linspace(1, N, N)
  y_label = np.array([CON]*N)

  plt.plot(x_label, x[0], color='r', linewidth=1.0, linestyle='--')
  plt.plot(x_label, y_label, color='g', linewidth=1.0, linestyle='-')
  plt.plot(x_label, y[0], color='b', linewidth=1.0, linestyle=':')
  plt.show()
