import numpy as np
import matplotlib.pyplot as plt

def get_viscuit(filename, T, rho, dt, V, rng=600, fmt="binary"):

    """
    Function to get_viscuit from a time history of shear stress
        Filename - containing Pxy virial stress tim e history (binary or ascii format)
        T - temperature
        rho - density
        V - Volume of domain
        rng - Time to integrate for viscuit
        fmt - Specify if filename is Binary or ascii
    """

    if fmt is "binary":
        d = np.fromfile(open(filename, 'rb'), dtype=np.float64)  
    elif fmt is "ascii":
        d = np.genfromtxt(filename)
    else:
        raise AttributeError("Error - fmt must be binary or ascii")

    viscuit = []
    for shift in range(0, d.shape[0]-rng):
        autocorrel = d[shift]*d[shift:shift+rng]
        viscuit.append(np.trapz(autocorrel[:], dx=dt))

    return (V/T)*np.array(viscuit)


if __name__ == "__main__":

    rho = 0.8442
    T = 0.722
    V = 10409.86
    dt = 0.005
    viscuit = get_viscuit("./data/Pxy_hist.b", T, rho, dt, V, rng=600, fmt="binary")

    #Show the viscuit distribution
    h, b = np.histogram(viscuit, 100, density=True)
    b = 0.5*(b[:-1] + b[1:])
    plt.plot(b, h)
    plt.show()

    #Compare viscosity from Green Kubo and 1st moment of viscuit PDF
    print("Viscosity  = ", np.mean(viscuit), "\int xP(x) dx = ", np.trapz(h*b, b))


