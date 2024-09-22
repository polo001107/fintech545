import numpy as np
import time
from scipy.linalg import norm

def chol_psd(a):
    n = a.shape[0]
    root = np.zeros_like(a)

    for j in range(n):
        s = 0.0
     
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        temp = a[j, j] - s
        
        if 0 >= temp >= -1e-8:
            temp = 0.0
        
        root[j, j] = np.sqrt(temp)

        if root[j, j] != 0.0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir

    return root

def generate_non_psd_matrix(n):
    np.random.seed(42)  
    A = np.random.randn(n, n)
    A = (A + A.T) / 2  
    np.fill_diagonal(A, 1)  
    return A

def higham_psd(a, max_iter=100):
    n = a.shape[0]
    W = np.identity(n)
    delta_S = np.zeros_like(a)
    Y = np.copy(a)
    for _ in range(max_iter):
        R = Y - delta_S
        X = np.copy(R)
        vals, vecs = np.linalg.eigh(X)
        vals[vals < 0] = 0
        X = vecs @ np.diag(vals) @ vecs.T
        delta_S = X - R
        Y = W @ X @ W
    return Y


def near_psd(a, epsilon=0.0):
    n = a.shape[0]
    out = np.copy(a)
    invSD = None
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / np.sqrt(vecs @ np.diag(vals) @ vecs.T)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out

def test_and_compare(n):
    
    A = generate_non_psd_matrix(n)

    start_time = time.time()
    near_psd_result = near_psd(A)
    near_psd_time = time.time() - start_time
    start_time = time.time()
    higham_psd_result = higham_psd(A)
    higham_time = time.time() - start_time
    frobenius_near_psd = norm(A - near_psd_result, 'fro')
    frobenius_higham_psd = norm(A - higham_psd_result, 'fro')
    print(f"Matrix size: {n}x{n}")
    print(f"near_psd runtime: {near_psd_time:.5f} seconds, Frobenius norm: {frobenius_near_psd:.5f}")
    print(f"Higham runtime: {higham_time:.5f} seconds, Frobenius norm: {frobenius_higham_psd:.5f}")
    
    return near_psd_time, higham_time, frobenius_near_psd, frobenius_higham_psd

matrix_sizes = [100, 200, 500]
for size in matrix_sizes:
    test_and_compare(size)
