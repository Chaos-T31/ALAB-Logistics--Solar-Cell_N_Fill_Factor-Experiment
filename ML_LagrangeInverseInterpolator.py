# ***************************************************************************************************************************************************************************************************** #
# ************************************************************ For Inverse Interpolating the Values of Current obtained @SolarCell.py ***************************************************************** #
# ***************************************************************************************************************************************************************************************************** #
# -@AmiLab


# For calculating the Pochammer Products for the Numerator and Denominator of the Terms of the Interpolating Polynomial-
def prod(y, Y, indx):
    P = 1
    for i in range(len(Y)):
        if i != indx:
            P *= y - Y[i]
        else:
            P *= 1
    return P

# Testing-
P = prod(10, [1, 2, 3], 2)
print(P)


# For calculating the Interpolation Terms(without the x' coefficient)
def interpoleTerms(y, Y, i):
    return prod(y, Y, i) / prod(Y[i], Y, i)

# Testing-
T = interpoleTerms(10, [1, 2, 3], 2)
print(T)


# For finding the Interpolating values at the given y-data points (y)i's
def invInterpolator(cords, y):
    S, x = 0, []
    for i in range(len(cords['y'])):
        x.append(i)
        S += interpoleTerms(y, cords['y'], i) * cords['x'][i]
    return S


# Testing-
y = invInterpolator({'x': [1, 2, 3], 'y': [3, 2, 1]}, 10)
print(y)



# *********************************************************************************** Section ends here ************************************************************************************************ #


# ******************************************************************************* Continuation @SolarCell.py ******************************************************************************************* #



