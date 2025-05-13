#
# TITLE: Computational Electronics
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: https://www.youtube.com/playlist?list=PLJtAfFg1nIX9dsWGnbgFt2dqvwXVRSVxm
#


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


#
# CLASS: Fundamental Constants
#

class FC:
    def __init__(self):
        #
        self.q = 1.602192e-19           # in C
        self.kb = 1.380662e-23          # in J/K
        self.ep0 = 8.854187817e-12      # in F/m
        self.mu0 = 4.0e-12 * np.pi      # in H/m
        self.h = 6.62617e-34            # in J/s
        self.hb = self.h / (2*np.pi)    # in J/s
        self.m0 = 9.109534e-31          # in Kg

        # intrinsic charge density
        self.ni = 1e16                  # in m^-3


#
# CLASS: NEWTON METHOD
#

class NEWTON:
    def __init__(self):
        #
        self.fc = FC()
        # variable
        self.phi = sp.symbols('phi')
        self.f = sp.Function('f')(self.phi)
        # derivative
        self.df_dphi = self.f.diff(self.phi)


    def set_expression(self, op_temp, dopant_density):
        # user input
        self.V_T = self.fc.kb * op_temp / self.fc.q
        # user input
        self.expr  = self.fc.ni * sp.exp(  self.phi / self.V_T )
        self.expr -= self.fc.ni * sp.exp( -self.phi / self.V_T )
        self.expr -= dopant_density
        
        # derivative
        self.dexpr_dphi = self.df_dphi.subs(self.f, self.expr).doit()
        # delta phi
        self.delta_phi = -self.expr / self.dexpr_dphi


    def newton_method(self, phi_old):
        #
        vals = {self.phi: phi_old}
        #
        eval_delta_phi = self.delta_phi.evalf(subs=vals)

        #
        return (phi_old + eval_delta_phi)

    



#
# MAIN
#

dopant_density_array = [1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24]
electric_potential_array = []
iteration_array = []

newton_method = NEWTON()

for index, dopant_density in enumerate(dopant_density_array):
    #
    error = 1.0
    iteration = 0

    #
    if index == 0:
        newton_method.set_expression(op_temp=300, dopant_density=dopant_density)
        phi = 0.0
        while error > 1e-3:
            phi_prev = phi
            phi = newton_method.newton_method(phi_prev)
            error = np.abs(phi - phi_prev)
            iteration += 1
        iteration_array.append(iteration)
        electric_potential_array.append(phi)

    else:
        newton_method.set_expression(op_temp=300, dopant_density=dopant_density)
        phi = electric_potential_array[-1]
        while error > 1e-3:
            phi_prev = phi
            phi = newton_method.newton_method(phi_prev)
            error = np.abs(phi - phi_prev)
            iteration += 1
        iteration_array.append(iteration)
        electric_potential_array.append(phi)
            
    #
    print('%i, %.6f' % (iteration, phi) )

