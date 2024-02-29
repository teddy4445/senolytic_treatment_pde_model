#!/usr/bin/env python

# library imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def main():
    """
    Single entry point for the project
    """
    # Define parameters
    num_points = 15*24
    r = np.linspace(0, 15, num_points)
    dr = r[1] - r[0]

    # Initial conditions
    D_initial = 2e-4
    T_initial = 5e-4
    E_initial = 1.2e-3
    W_initial = 3.6e-4
    V_initial = 3e-9
    I_initial = 4e-10
    Cs_initial_coeff = 0.2
    phi = 4.06

    # Compute Cs initial condition based on the given coefficient
    C0_initial = (phi - D_initial - T_initial - E_initial)/(1 + Cs_initial_coeff)

    # Define system of ODEs
    def system_of_odes(t, Y):
        C, Cs, D, T, E, W, I, V = y

        # Define the parameters
        delta = 8.64e-3
        delta_W = 0.8
        delta_I = 6.05e-2
        delta_V = 8.64e-2
        d_C = 0.1
        d_Cs = 0.92
        d_D = 0.1
        d_T = 0.18
        d_E = 0.69
        d_W = 1.04
        d_I = 1.38
        d_V = 12.6
        C_0 = 0.8
        E_0 = 5e-3
        D_0 = 2e-5
        T_0 = 2e-4
        W_0 = 4.65e-4
        W_star = 1.69e-4
        V_0 = 3.65e-10
        chi = 0.8
        K_C = 0.4
        K_D = 4e-4
        K_T = 1e-3
        K_E = 2.5e-3
        K_W = 1.69e-4
        K_I = 8e-10
        K_V = 7e-8
        lambda_CW = 1.7
        lambda_CCs = 0.92
        lambda_D = 4
        lambda_T = 1.8
        lambda_EV = 1.87e7
        lambda_WE = 7.4e-2
        lambda_ID = 5.52e-6
        lambda_VW = 14.7e-6
        mu_TC = 500
        d_TI = 2.76
        d_EV = 25.2
        lambda_s = 5
        T_hat = 2e-3
        E_hat = 5e-3
        alpha = 1
        beta = 1
        gamma = 1

        # Define lambda_W
        lambda_W = lambda_CW * (W / W_0) if W <= W_0 else lambda_CW

        # Define lambda_E
        lambda_E = lambda_EV * (V - V_0) if V >= V_0 else 0.0

        # Define lambda_V
        if 0 <= W <= W_star:
            lambda_V = lambda_VW * (W / W_star)
        elif W_star < W <= W_0:
            lambda_V = 1.0 - 0.7 * ((W - W_star) / (W_0 - W_star))
        else:
            lambda_V = 0.3

            # Define the PDEs
            dC_dt = - delta * np.dot(u, C) + lambda_W * C * (1 - C / C_0) - mu_TC * T * C - mu_PC * C * P - d_C * C
            dCs_dt = - delta * np.dot(u, Cs) + lambda_CCs * C - mu_FC_s * Cs * F + lambda_PC_s * C * P - d_Cs * Cs
            dD_dt = - delta * np.dot(u, D) + lambda_D * D_0 * C / (K_C + C) - d_D * D
            dT_dt = - delta * np.dot(u, T) + lambda_T * T_0 * I / (K_I + I) - mu_PT * T * P - d_T * T
            dE_dt = - delta * np.dot(u, E) + lambda_E * E * (1 - E / E_0) - np.dot(chi * E,
                                                                                   np.dot(grad(V), grad(V))) - d_E * E
            dW_dt = - delta_W * np.dot(grad(W), grad(W)) + lambda_WE * E - d_W * W
            dI_dt = - delta_I * np.dot(grad(I), grad(I)) + lambda_ID * D - d_TI * I * T / (K_T + T) - d_I * I
            dV_dt = - delta_V * np.dot(grad(V), grad(
                V)) + lambda_V * C + lambda_s * lambda_V * Cs - mu_FV * V * F - d_EV * V * E / (K_E + E) - d_V*V

            return [dC_dt, dCs_dt, dD_dt, dT_dt, dE_dt, dW_dt, dI_dt, dV_dt]

    # Initial conditions vector
    Y0 = [D_initial, T_initial, E_initial, W_initial, V_initial, I_initial, Cs_initial_coeff * C0_initial,
          C0_initial]

    # Time span
    t_span = (0, 15)

    # Solve the system of ODEs
    sol = solve_ivp(system_of_odes, t_span, Y0, t_eval=np.linspace(t_span[0], t_span[1], num_points))

    # Plot the solutions
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 3, 1)
    plt.plot(sol.t, sol.y[0], label='D')
    plt.xlabel('Time')
    plt.ylabel('D')

    plt.subplot(3, 3, 2)
    plt.plot(sol.t, sol.y[1], label='T')
    plt.xlabel('Time')
    plt.ylabel('T')

    plt.subplot(3, 3, 3)
    plt.plot(sol.t, sol.y[2], label='E')
    plt.xlabel('Time')
    plt.ylabel('E')

    plt.subplot(3, 3, 4)
    plt.plot(sol.t, sol.y[3], label='W')
    plt.xlabel('Time')
    plt.ylabel('W')

    plt.subplot(3, 3, 5)
    plt.plot(sol.t, sol.y[4], label='V')
    plt.xlabel('Time')
    plt.ylabel('V')

    plt.subplot(3, 3, 6)
    plt.plot(sol.t, sol.y[5], label='I')
    plt.xlabel('Time')
    plt.ylabel('I')

    plt.subplot(3, 3, 7)
    plt.plot(sol.t, sol.y[6], label='Cs')
    plt.xlabel('Time')
    plt.ylabel('Cs')

    plt.subplot(3, 3, 8)
    plt.plot(sol.t, sol.y[7], label='C')
    plt.xlabel('Time')
    plt.ylabel('C')

    plt.tight_layout()
    plt.savefig("results.pdf", dpi=400)
    plt.show()
    plt.close()


if __name__ == "__main__":
    # run the server
    main()
