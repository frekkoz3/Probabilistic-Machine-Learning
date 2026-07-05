import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal

np.random.seed(7)

# =====================================================
# Generate synthetic data
# =====================================================

N = 500

true_means = np.array([
    [2, 2],
    [3.5, 3]
])

true_covs = np.array([
    [[1.2, 0.5],
     [0.5, 0.8]],

    [[1.0, -0.4],
     [-0.4, 1.4]]
])

X = np.vstack([
    np.random.multivariate_normal(true_means[0], true_covs[0], N//2),
    np.random.multivariate_normal(true_means[1], true_covs[1], N//2)
])

# =====================================================
# EM Initialization
# =====================================================

K = 2

means = np.random.uniform(-2, 7, (K,2))
covs = np.array([np.eye(2)*2 for _ in range(K)])
weights = np.ones(K)/K

# =====================================================
# Mesh for Gaussian contours
# =====================================================

xmin, ymin = X.min(axis=0)-1
xmax, ymax = X.max(axis=0)+1

xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, 120),
    np.linspace(ymin, ymax, 120)
)

grid = np.dstack((xx,yy))

# =====================================================
# Plot
# =====================================================

fig, ax = plt.subplots(figsize=(8,8))


def em_step():

    global means, covs, weights

    # ---------- E STEP ----------

    responsibilities = np.zeros((len(X),K))

    for k in range(K):

        responsibilities[:,k] = (
            weights[k] *
            multivariate_normal.pdf(
                X,
                mean=means[k],
                cov=covs[k]
            )
        )

    responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    # ---------- M STEP ----------

    Nk = responsibilities.sum(axis=0)

    weights = Nk/len(X)

    for k in range(K):

        means[k] = (
            responsibilities[:,k,None] * X
        ).sum(axis=0)/Nk[k]

        diff = X-means[k]

        covs[k] = (
            responsibilities[:,k,None,None]
            * np.einsum("ni,nj->nij",diff,diff)
        ).sum(axis=0)/Nk[k]

    # ---------- Log likelihood ----------

    likelihood = np.zeros((len(X),K))

    for k in range(K):

        likelihood[:,k] = (
            weights[k]
            * multivariate_normal.pdf(
                X,
                means[k],
                covs[k]
            )
        )

    ll = np.sum(np.log(likelihood.sum(axis=1)))

    return responsibilities, ll


# =====================================================
# Animation
# =====================================================

history = []


def update(frame):

    ax.clear()

    gamma, ll = em_step()

    history.append(ll)

    colors = gamma[:,0]

    ax.scatter(
        X[:,0],
        X[:,1],
        c=colors,
        cmap="coolwarm",
        s=18,
        alpha=.7
    )

    # Draw Gaussian contours

    for k in range(K):

        rv = multivariate_normal(
            means[k],
            covs[k]
        )

        zz = rv.pdf(grid)

        ax.contour(
            xx,
            yy,
            zz,
            levels=5,
            linewidths=2
        )

        ax.scatter(
            means[k,0],
            means[k,1],
            marker="X",
            s=250,
            edgecolors="black"
        )

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

    ax.set_title(
        f"Iteration {frame+1}\n"
        f"Log-Likelihood = {ll:.2f}"
    )

    if len(history)>2:
        if abs(history[-1]-history[-2]) < 1e-3:
            ani.event_source.stop()


ani = FuncAnimation(
    fig,
    update,
    frames=30,
    interval=800,
    repeat=False
)

plt.show()