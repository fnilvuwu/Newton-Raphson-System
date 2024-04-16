import numpy as np
import math

def newton_raphson_system(f, Df, x0, tol=1e-5, max_iter=100):
    """
    Solves a system of non-linear equations using Newton-Raphson method.

    Args:
        f: A function that takes a vector x and returns a vector of function values f(x).
        Df: A function that takes a vector x and returns the Jacobian matrix of f(x).
        x0: An initial guess for the solution.
        tol: The tolerance for convergence.
        max_iter: The maximum number of iterations.

    Returns:
        A vector containing the approximate solution or None if convergence is not achieved.
    """
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        f_val = np.array(f(x), dtype=float)
        if np.linalg.norm(f_val) < tol:
            return x
        J = Df(x)
        try:
            dx = np.linalg.solve(J, -f_val)
        except np.linalg.LinAlgError:
            print("Singular Jacobian encountered. Failed to converge.")
            return None
        x += dx
    print("Failed to converge after", max_iter, "iterations.")
    return None

# Define the system of equations
def f(x):
    x1, x2 = x
    return [
        (x1 - x2)**3*x2 + 1,
        x1*x2**2 - math.sin(x1)
    ]

# Define the Jacobian of the system
def Df(x):
    x1, x2 = x
    return np.array([
        [3*(x1-1)**2*x2, (x1-1)**3],
        [x2**2 - math.cos(x1), 2*x1*x2]
    ])

# Initial guess
x0 = [1.5, 0.5]

# Print the initial guess
print("Initial guess:", x0)

# Print the output of f(x0)
print("f(x0) =", f(x0))

# Print the output of Df(x0)
print("Df(x0) =", Df(x0))

# Solve the system of equations
solution = newton_raphson_system(f, Df, x0)

# Print the solution
if solution is not None:
    print("Solution:", solution)

    # Check if answer is correct
    x1, x2 = solution
    print("f1(x) =", x1**2 + x2 - 2*x1 - 0.4)
    print("f2(x) =", x2*2 + 3*x1*x2 - 2*x1**3 - 2*x1 - 0.4)
