import numpy as np
import math

def protected_exp(x):
    # Overflow condition
    if x > 700:
        return math.exp(700)  
    # Underflow condition
    elif x < -700:
        return math.exp(-700)
    else:
        return math.exp(x)

def protected_div(a, b):
    if abs(b) < 1e-12:
        return 1.0
    return a / b

def protected_sqrt(x):
    if x < 0:
        return np.sqrt(abs(x))
    return np.sqrt(x)

def protected_log10(x):
    if x <= 0:
        return 0.0
    return np.log10(x)

def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])

def f2(x: np.ndarray) -> np.ndarray:
    return (
        (
            protected_exp((x[0] * 4.865) * protected_exp(x[1])) * protected_div((-1.527 - 9.650) * (x[2] + x[1]), x[2]) * np.abs(5.823 + x[0])
        ) -
        (protected_div((x[2] - (-2.865)),(-9.485 + 5.293)))
    )

def f3(x: np.ndarray) -> np.ndarray:
    return (
        (x[0] * x[0]) + 
        (x[2] * -4.816) + 
        (x[2] - (-3.753)) + 
        (x[0] * x[0]) + 
        (protected_log10((2.369 + x[1]) - x[1]) + (x[1] * x[1] * -x[1]))
    )

def f4(x: np.ndarray) -> np.ndarray: 
    return (
        (np.cos(x[1]) * 6.830) + 
        protected_sqrt(protected_sqrt((-9.009 + x[0]) + np.abs(x[1]))) + 
        protected_sqrt(protected_sqrt(-8.538))
    )

def f5(x: np.ndarray) -> np.ndarray:
    return (
        protected_div((
            (protected_log10(x[1]) * -9.013) * 
            ((0.309 - (-6.337)) - np.sin(x[0])) * 
            np.cos(protected_div(x[0],x[0]) * protected_div(-3.695,x[0]))
        ), (protected_exp(protected_div(-5.465,-0.213)) - (-3.879)))
    )

def f6(x: np.ndarray) -> np.ndarray:
    return (
        ((1.396 * x[1]) + protected_div(x[0],-1.418) - protected_div(x[1],-3.781)) -
        (
            protected_exp(-9.845) * (6.017 - x[0]) * 
            ((x[1] - x[0]) * ((-4.666 + x[0]) + (3.425 * -1.858))) * 
            (8.596 - x[0])
        )
    )

def f7(x: np.ndarray) -> np.ndarray:
    return (
        (np.abs(4.439) * protected_exp(x[1] * x[0])) + 
        (
            ((np.abs(x[0]) * np.abs(x[1])) + (np.abs(x[0]) * np.sin(x[1]))) *
            np.cos((x[1] - x[0]) * (x[0] + -7.844)) *
            protected_exp(x[1] * x[0])
        )
    )

def f8(x: np.ndarray) -> np.ndarray:
    return (
        np.abs(
            x[5] * 
            (9.915 * x[5]) * 
            (protected_div(x[4],x[4]) * (x[5] + x[5]))
        ) * x[5]
    )