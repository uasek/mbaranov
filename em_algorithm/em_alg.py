import numpy as np
from scipy.signal import fftconvolve
from scipy.special import softmax, xlogy


# --- help functions ---
def _face_area_conv(X, F, B):
    H, W, K = X.shape
    h, w = F.shape

    conv = fftconvolve(
        in1=X,
        in2=F[::-1, ::-1].reshape(h, w, 1),
        axes=(0, 1),
        mode='valid',
    )

    XB = -X * B.reshape(H, W, 1) + 0.5 * B.reshape(H, W, 1) ** 2
    ones = np.ones((h, w, 1))

    conv += fftconvolve(
        XB,
        ones,
        mode='valid',
        axes=(0, 1)
    )

    return conv

def _sums_B_F(X, F, B):
    sums = np.sum(
        (X - B.reshape(*B.shape, 1)) ** 2,
        axis=(0, 1)
    )

    sums += np.sum(F ** 2)

    return sums

def _transform_q(q, height, width):
    _, K = q.shape

    q_slice = np.c_[q.T, np.arange(K)]
    new_q = np.zeros((height, width, K))

    new_q[q_slice[:, 0], q_slice[:, 1], q_slice[:, 2]] = 1

    return new_q
# --- ### ---

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    sum_{ij} log p(X_k(i, j) | d, theta) 

    Parameters
    ----------
    X : array, shape (H, W, K)  # ну блин, данные W x H, забьем пока на это
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """

    H, W, K = X.shape

    # sum of squared deviations
    log_prob = _sums_B_F(X, F, B).reshape(1, 1, K) - 2 * _face_area_conv(X, F, B)
    log_prob /= -2 * s ** 2
    log_prob += -0.5 * H * W * np.log(2 * np.pi * s ** 2)

    return log_prob

def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """

    H, W, K = X.shape
    h, w = F.shape

    if use_MAP == True:
        q = _transform_q(q, H - h + 1, W - w + 1)

    logp_X_d = calculate_log_probability(X, F, B, s)

    lb = np.sum(logp_X_d * q)

    lb += np.sum(
        xlogy(q, A.reshape(*A.shape, 1))
    )

    lb -= np.sum(
        xlogy(q, q)
    )

    return lb

def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """

    H, W, K = X.shape
    conv = _face_area_conv(X, F, B)

    q = np.log(A, where=(A != 0.0)).reshape(*A.shape, 1) + conv / s ** 2
    q -= np.max(q, axis=(0, 1), keepdims=True)

    if use_MAP:
        argmax = np.argwhere(
            q == np.max(q, axis=(0, 1)).reshape(1, 1, K)
        )

        # sort by the third col
        argmax = argmax[np.argsort(argmax[:, 2]), :]

        return argmax[:, :2].T

    else:
        return softmax(q, axis=(0, 1))

def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """

    H, W, K = X.shape

    if use_MAP:
        q = _transform_q(q, H - h + 1, W - w + 1)

    # optimize by А
    A_new = np.sum(q, axis=2) / K

    # optimize by F
    F_new = fftconvolve(
        X,
        q[::-1, ::-1, :],
        mode='valid',
        axes=(0, 1)
    ).sum(2) / K

    # optimize by B
    alpha = 1 - fftconvolve(
        q,
        np.ones((h, w, 1)),
        mode='full',
        axes=(0, 1)
    )

    B_new = np.sum(X * alpha, axis=2) / np.sum(alpha, axis=2)

    # optimize by s
    s_new = np.sum(
        _sums_B_F(X, F_new, B_new)
    )

    s_new += -2 *np.sum(
        q * _face_area_conv(X, F_new, B_new)
    )

    s_new /= H * W * K
    s_new = np.sqrt(s_new)

    return F_new, B_new, s_new, A_new

def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step); 
        number_of_iters is actual number of iterations that was done.
    """

    H, W, K = X.shape

    lb_new = -np.inf
    LL = []

    if F is None:
        F = np.random.randn(h, w)

    if B is None:
        B = np.random.randn(H, W)

    if s is None:
        scale = np.std(X)
        s = np.random.exponential(scale=scale, size=1)[0]

    if A is None:
        A = np.random.rand(H - h + 1, W - w + 1)
        A /= A.sum()
    
    for i in range(max_iter):

        q = run_e_step(X, F, B, s, A, use_MAP=use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP=use_MAP)
        lb_old = lb_new
        lb_new = calculate_lower_bound(X, F, B, s, A, q, use_MAP=use_MAP)
        LL.append(lb_new.copy())

        if lb_new - lb_old < tolerance:
            break

    return F, B, s, A, LL  

def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """

    L = -np.inf
    
    for i in range(n_restarts):

        F, B, s, A, LL = run_EM(
            X, h, w, tolerance=tolerance, max_iter=max_iter, use_MAP=use_MAP
            )

        if LL[-1] > L:

            F_best = F
            B_best = B
            s_best = s
            A_best = A
            L = LL[-1]

    return F_best, B_best, s_best, A_best, L
        
