# ***************************************************************************************************************************************************************************************************** #
# ************************************************************ 14.	To study the characteristics of a solar cell and find the fill factor. ************************************************************ #
# ***************************************************************************************************************************************************************************************************** #
# -@AmiLab


'''
Note-
    - **DATA VALIDATION EXCLUDED FOR BEING CHECKED AT THE TIME OF DATA INPUT**
    - All Testings have been logged into the terminal for future debuggings.
    - The Length of Jib, Tie and Post is/are considered to be in the same unit system.
'''


# ********************************************************************** Argument / Variable Declaration (for Testing purposes) ********************************************************************** #



n = 7                                                                                                      # The Total Number of Observations been performed
divs = {'V': [50, 5.5, 4, 4.5, 6, 2, 1], 'I': [.6, .7, 1.2, 1.0, .6, 1.9, 2.1]}                            # The Divison Readings of the Voltmeter and Ammeter
msr = {'V': 100, 'I': 50}                                                                                  # Main Scale Readings(MSRs) of the Voltmeter and the Ammeter
total_divs = {'V': 10, 'I': 10}                                                                            # Total Number of divisions on the Main Scale of the Voltmeter and the Ammeter



# **************************************************************************************** Section ends here **************************************************************************************** #


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #



# ******************************************************************* Calculation of Fill Factor of Solar Cell ******************************************************************* #



import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from ML_LagrangeInverseInterpolator import invInterpolator
from scipy.interpolate import InterpolatedUnivariateSpline                               # For 2 paths for performing Interpolation. Helpful in tallying and verification.
from ML_RadialBasisFunction_RBF_Interpolator import RBF_Interpolator                     # For 2 paths for performing Interpolation. Helpful in tallying and verification.


print('MSR =', msr)
print('Total Number of Divisions =', total_divs)


def cal_LC(msr, total_divs):
    global V, I
    V, I = list(msr.keys())[0], list(msr.keys())[1]
    return {str(V): msr[V] / total_divs[V], str(I): msr[I] / total_divs[I]}

# Testing-
lc = cal_LC(msr, total_divs)
print('Least Count (L.C.) of the Voltmeter and Ammeter =', lc)


def calcMetersReadings(divs, lc, n):
    return {str(V): [divs[str(V)][i] * lc[str(V)] for i in range(n)], str(I): [divs[str(I)][i] * lc[str(I)] for i in range(n)]}

# Testing-
readings = calcMetersReadings(divs, lc, n)
print('V-I Readings =', readings)


def calPower(readings, n):
    return [readings[str(V)][i] * readings[str(I)][i] for i in range(n)]

# Testing-
pow = calPower(readings, n)
print('Powers =', pow)


def plot_VI_CharacteristicsCurve(readings, n):
    readings_pairs = [(readings[str(V)][i], readings[str(I)][i]) for i in range(n)]
    readings_pairs.sort(key = lambda x: x[0])

    print('(V, I) =', readings_pairs)

    x = np.array([readings_pairs[i][0] for i in range(n)])
    y = np.array([readings_pairs[i][1] for i in range(n)])

    x_new = np.linspace(0, 40, 14)
    bspline = interpolate.make_interp_spline(x, y)
    y_new = bspline(x_new)

    max_pow = np.amax(np.array([x_new[i] * y_new[i] for i in np.arange(x_new.shape[0])]))

    x_max, y_max = [], []
    for i in np.arange(x_new.shape[0]):
        if x_new[i] * y_new[i] == max_pow:
            x_max.append(x_new[i])
            y_max.append(y_new[i])
            break

    plt.plot(x_new, y_new, color = 'b')
    plt.title("Solar Cell's Potential Difference(V) v/s. Current(I) Characteristics.")
    plt.xlabel('V')
    plt.ylabel('I')

    plt.show()
    return [x_max, y_max, max_pow, x, y, x_new, y_new]

# Testing-
x_max, y_max, max_pow, x, y, x_new, y_new = plot_VI_CharacteristicsCurve(readings, n)
print(f'x_max = {x_max}\ny_max = {y_max}\nmax_pow = {max_pow}')
print(f'x = {x}\ny = {y}')
print(f'x_new = {x_new}\ny_new = {y_new}')

XnY = {'x': x_new, 'y': y_new}
y0 = invInterpolator({'x': x_new, 'y': y_new}, 0)
print(y0)


def interpolator(XnY, x):
    return InterpolatedUnivariateSpline(XnY['x'], XnY['y'])(x)

# Testing-
x0 = interpolator(XnY, 0)
print(x0)

def RBF_interpolator(XnY, x):
    return RBF_Interpolator(XnY, x)

# Testing-
x1 = RBF_interpolator(XnY, 0)
print(x1)


def calFillFactor(max_pow, x0, y0):
    return abs(max_pow / (x0 * y0))

# Testing-
fill_factor = calFillFactor(max_pow, x0, y0)
print('Fill Factor of the Solar Cell =', fill_factor)



# ********************************************************************************* Section ends here *********************************************************************************************** #


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #




