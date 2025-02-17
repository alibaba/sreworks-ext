import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import scipy as scp
normalize = lambda p: p/np.sum(p)
def opt(N,t,a,b):
    vmin = .02
    a = normalize(a + (np.max(a)+vmin) * vmin)
    b = normalize(b + (np.max(b)+vmin) * vmin)
    epsilon = (.03) ** 2

    [Y, X] = np.meshgrid(t, t)
    K = np.exp(-(X - Y) ** 2 / epsilon)

    v = np.ones(N)
    niter = 100
    Err_p = np.zeros(niter)
    Err_q = np.zeros(niter)
    for i in range(niter):
        u = a / (np.dot(K, v))
        r = v * (np.dot(K, u))
        Err_q[i] = np.linalg.norm(r - b, ord=1)
        v = b / (np.dot(K, u))
        s = u * (np.dot(K, v))
        Err_p[i] = np.linalg.norm(s - a, ord=1)
    P = np.dot(np.dot(np.diag(u), K), np.diag(v))
    #plt.figure(figsize=(5, 5))
    #plt.imshow(np.log(P + 1e-5))
    #plt.axis('off')
    #plt.show()
    return P

if __name__=="__main__":
    N = 200
    t = np.arange(0, N)/N

    Gaussian = lambda t0, sigma: np.exp(-(t-t0)**2/(2*sigma**2))



    sigma = .06
    a = Gaussian(.25, sigma)
    b = Gaussian(.8, sigma) + 3 * Gaussian(.6, sigma) + Gaussian(0.4, sigma)



    plt.figure(figsize = (10,7))

    plt.subplot(2, 1, 1)
    plt.bar(t, a, width = 1/len(t), color = "darkblue")
    plt.subplot(2, 1, 2)
    plt.bar(t, b, width = 1/len(t), color = "darkblue")

    opt(N,t,a,b)