import numpy as np

np.random.seed(1)

# -------------------------------------------------------
# Synthetic observations
# -------------------------------------------------------

# Number of heads in 10 tosses

observations = np.array([5,9,8,4,7])

n = 10

# Initial guesses

thetaA = 0.6
thetaB = 0.5

for iteration in range(10):

    # ---------- E STEP ----------

    gamma = []

    for h in observations:

        pA = thetaA**h * (1-thetaA)**(n-h)
        pB = thetaB**h * (1-thetaB)**(n-h)

        g = pA/(pA+pB)

        gamma.append(g)

    gamma = np.array(gamma)

    # ---------- M STEP ----------

    expected_heads_A = np.sum(gamma * observations)
    expected_tails_A = np.sum(gamma * (n-observations))

    expected_heads_B = np.sum((1-gamma)*observations)
    expected_tails_B = np.sum((1-gamma)*(n-observations))

    thetaA = expected_heads_A / (expected_heads_A+expected_tails_A)
    thetaB = expected_heads_B / (expected_heads_B+expected_tails_B)

    print(f"Iteration {iteration+1}")
    print(f"thetaA = {thetaA:.3f}")
    print(f"thetaB = {thetaB:.3f}")
    print()