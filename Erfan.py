import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.image as mpimg
import cv2
import os.path

B = np.zeros((2, 2, 2, 2))
basis = 1/sqrt(2)*np.array([[1, 1], [1, -1]])
for i in range(0, 2):
    for j in range(0, 2):
        print("{}, {}:".format(i, j))
        B[:, :, i, j] = np.outer(basis[i, :], basis[j, :])
        print( B[:, :, i, j])
        print()

# DCT of f
f = np.array([[3, 4], [2, 1]])
alpha = np.zeros([2, 2])
for i in range(0, 2):
    for j in range(0, 2):
        BB = B[:, :, i, j]
        alpha[i, j] = np.inner(np.reshape(BB, -1), np.reshape(f, -1))
print("alpha:", alpha)
print()

# DCT matrix for arbitrary N
N = 8
C = np.zeros([N, N])
C[0, :] = 1/sqrt(N)
for k in range(1, N):
    for m in range(0, N):
        C[k, m] = sqrt(2/N) * np.cos(np.pi*k*(m+0.5)/N)

# Plot Basis vector
x = list(range(N))
for i in range(0, N):
    plt.subplot(4, 2, i+1), title('Basis %i' % (i+1))
    plt.subplots_adjust(hspace=1)
    plt.plot(x, C[i, :])
plt.show()

# Checking Orthonormality
ortho_Ch = np.zeros([N, N])
for i in range(0, N):
    for j in range(0, N):
        ortho_Ch[i, j] = np.inner(C[i, :], C[j, :])
print(ortho_Ch) # Try and Print the matrix in 2 decimal digits

# Checking inverse=transpose
diff_Tr_In = np.linalg.norm(np.linalg.inv(C)-np.matrix.transpose(C))
print('C inv - C Tr difference %0.2f' % diff_Tr_In)

#  DCT is unitary
Prod = np.matmul(C, np.matrix.transpose(C))
# alternative: Prod = C @ np.matrix.transpose(C)
Diff = Prod - np.identity(N)
print('Diff= %0.2f' % np.linalg.norm(Diff))

# Do - Undo DCT

#x = cv2.imread('512.gif',0) # cv2 can't read .gif!
x = mpimg.imread('lenna256.gif')
Her_eye = x[130:138, 130:138, 0] #mpimg.imread returns a 4-page tensor for a grayscale image for some reason!!!
DCT_coeffs = C @ Her_eye @ np.matrix.transpose(C)
plt.figure()
ll = plt.imshow(DCT_coeffs, cmap='gray')
plt.show()

Her_eye2 = np.matrix.transpose(C)@ DCT_coeffs@ C
plt.figure()
ll = plt.imshow(Her_eye2, cmap='gray')
plt.show()

# Chopping off higher DCT coeffs
DCT_coeffs2 = DCT_coeffs
for i in range(0,N):
    for j in range(0,N):
        if i+j > 10:
            DCT_coeffs[i,j] = 0
Her_eye3 = np.matrix.transpose(C)@ DCT_coeffs2@ C
plt.figure()
ll = plt.imshow(Her_eye3, cmap='gray')
plt.show()

DIFF = Her_eye - Her_eye3
plt.figure()
ll = plt.imshow(DIFF, cmap='gray')
plt.show()

# Bonus
x = mpimg.imread('MRA.png')
x0 = x
[N1, N2] = shape(x)
for i in range(64):
    for j in range(64):
        B = x[8*i:8*(i+1), 8*j:8*(j+1)]
        DCT = C@ B @ np.matrix.transpose(C)
        for k in range(0,8):
              for l in range(0,8):
                  if k+l> 0:
                      DCT[k, l] = 0
                      B = np.matrix.transpose(C) @ DCT @ C
                      x[8*i:8*(i+1), 8*j:8*(j+1)] = B
plt.imshow(x, cmap='gray')
plt.show()