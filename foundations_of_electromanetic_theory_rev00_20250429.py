#
# TITLE:
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: foundations of electromagnetic theory, 4th ed., Addision Wesley (1993) 
#


import sys
import time 
import numpy as np
#from numba import jit
import matplotlib.cm as cm
import matplotlib.pyplot as plt


#==============================================
# class ES2D
# example 3-5. potential in retangular region
#
# Poisson equation: ∇·( ε(x,y) * ∇φ(x,y) ) = -ρ(x,y),
# if ρ(x,y) = 0, then Laplace equation ∇·( ε(x,y) * ∇φ(x,y) ) = 0.0,
# if ε(x,y) = constant, then ∇·( ∇φ(x,y) ) = 0.0.
#
# discretization
# ( φ(i+1,j) + φ(i-1,j) - 2*φ(i,j) ) / dx**2 + ( φ(i,j+1) + φ(i,j-1) - 2*φ(i,j) ) / dy*2 = 0.0
# φ(i,j) * ( 2 / dx**2 + 2 / dy**2 ) = ( φ(i+1,j) + φ(i-1,j) ) / dx**2 + ( φ(i,j+1) + φ(i,j-1) ) / dy*2
# φ(i,j) = ( ( φ(i+1,j) + φ(i-1,j) ) / dx**2 + ( φ(i,j+1) + φ(i,j-1) ) / dy*2 ) / ( 2 / dx**2 + 2 / dy**2 )
#
# iterations
# φ(i,j,n+1) = ( ( φ(i+1,j,n) + φ(i-1,j,n) ) / dx**2 + ( φ(i,j+1,n) + φ(i,j-1,n) ) / dy*2 ) / ( 2 / dx**2 + 2 / dy**2 )
#
# overrelaxation for fast convergence
# φ(i,j,n+1) = w * ( ( φ(i+1,j,n) + φ(i-1,j,n) ) / dx**2 + ( φ(i,j+1,n) + φ(i,j-1,n) ) / dy*2 ) / ( 2 / dx**2 + 2 / dy**2 )
#                + (1 - w ) * φ(i,j,n) 
#

class ES2D:

    def __init__(self):
        # electric potential
        self.phi = []

        # speed of convergence
        self.iteration_no = 0


    def make_rectangular_box(self, geometry_info):
        #
        self.geometry_info = geometry_info

        # x info, y info
        self.x_info = self.geometry_info['x']
        self.y_info = self.geometry_info['y']

        # element length
        self.dx = self.x_info['length'] / self.x_info['elements']
        self.dy = self.y_info['length'] / self.y_info['elements']

        # x, y nodes
        self.x_nodes = self.x_info['elements'] + 1
        self.y_nodes = self.y_info['elements'] + 1
        
        # materials: 1 for dielectrics, 0 for metal
        self.mat = np.ones([self.x_nodes, self.y_nodes], dtype=float)

        # electric potential
        self.phi = np.zeros([self.x_nodes, self.y_nodes], dtype=float)

        # electric field
        self.ex = np.zeros([self.x_nodes-1, self.y_nodes], dtype=float)
        self.ey = np.zeros([self.x_nodes, self.y_nodes-1], dtype=float)
        self.em = np.zeros([self.x_nodes-1, self.y_nodes-1], dtype=float)


    def dirichlet_boundary_conditions(self, bias_conditions):
        #
        self.bias_conditions = bias_conditions

        #
        self.bc_v_left = self.bias_conditions['x']['left']
        self.mat[0,:] = 0.0
        self.phi[0,:] = self.bc_v_left

        #
        self.bc_v_right = self.bias_conditions['x']['right']
        self.mat[-1,:] = 0.0
        self.phi[-1,:] = self.bc_v_right

        #
        self.bc_v_upper = self.bias_conditions['y']['upper']
        self.mat[:,-1] = 0.0
        self.phi[:,-1] = self.bc_v_upper

        #
        self.bc_v_lower = self.bias_conditions['y']['lower']
        self.mat[:,0] = 0.0
        self.phi[:,0] = self.bc_v_lower


    def finite_difference_method(self, overrelaxation, conv_error):
        #
        print('iteration start time = %s' % time.ctime())
        
        #
        self.w = overrelaxation
        
        #
        self.x_coeff = self.w / self.dx**2 / ( 2.0 / self.dx**2 + 2.0 / self.dy**2 )
        self.y_coeff = self.w / self.dy**2 / ( 2.0 / self.dx**2 + 2.0 / self.dy**2 )
        self.conv_coeff = 1.0 - self.w
        
        #
        error = 1.0
        self.iteration_no = 0

        while error > conv_error:
            #
            prev_phi = self.phi.copy()
            
            # iteration
            for x_node_cnt in range(self.x_nodes):
                for y_node_cnt in range(self.y_nodes):
                    # material check: metal
                    if self.mat[x_node_cnt,y_node_cnt] == 0:
                        pass
                    # material check: dielectrics
                    else:
                        self.phi[x_node_cnt,y_node_cnt] = self.x_coeff * (self.phi[x_node_cnt+1,y_node_cnt] + self.phi[x_node_cnt-1,y_node_cnt]) + \
                                                          self.y_coeff * (self.phi[x_node_cnt,y_node_cnt+1] + self.phi[x_node_cnt,y_node_cnt-1]) + \
                                                          self.conv_coeff * self.phi[x_node_cnt,y_node_cnt]
            # error
            error = np.max( np.abs(prev_phi - self.phi) )

            # debugging
            if self.iteration_no % 10 == 0:
                output_string = 'iteration = %i, error = %.2e, time = %s' % (self.iteration_no, error, time.ctime())
                print(output_string)

            # iteration no.
            self.iteration_no += 1

        #
        print( 'w = %.2f, error = %.2e, iteration = %i ea' % (self.w, conv_error, self.iteration_no) )

        #
        print('iteration end time = %s' % time.ctime())


    def make_unit_cell_geometry(self, unit_cell_info):
        # user input
        coordi_x_lower_limit = unit_cell_info['coordi_x_limit']['lower']
        coordi_x_upper_limit = unit_cell_info['coordi_x_limit']['upper']
        coordi_x_elements = unit_cell_info['coordi_x_limit']['elements']
        coordi_x_upper_limit = unit_cell_info['coordi_x_limit']['upper']
        coordi_y_lower_limit = unit_cell_info['coordi_y_limit']['lower']
        coordi_y_upper_limit = unit_cell_info['coordi_y_limit']['upper']
        coordi_y_elements = unit_cell_info['coordi_y_limit']['elements']

        ponoa_alo_thk = unit_cell_info['ponoa_alo']['thk'][0]
        ponoa_box_thk = unit_cell_info['ponoa_box']['thk'][0]
        ponoa_ctn_thk = unit_cell_info['ponoa_ctn']['thk'][0]
        ponoa_tox_thk = unit_cell_info['ponoa_tox']['thk'][0]
        ponoa_pch_thk_major = unit_cell_info['ponoa_pch']['thk'][0]
        ponoa_pch_thk_minor = unit_cell_info['ponoa_pch']['thk'][1]

        ch_cut_cd = unit_cell_info['cut_cd']

        gate_major_cd = unit_cell_info['gate']['major_CD']
        gate_distortion = unit_cell_info['gate']['distortion']

        # calculations
        gate_minor_cd = gate_major_cd * gate_distortion

        onoa_thk = np.sum([ponoa_alo_thk, ponoa_box_thk, ponoa_ctn_thk, ponoa_tox_thk]) 
        pch_major_cd1 = gate_major_cd - 2.0 * onoa_thk
        pch_minor_cd1 = gate_minor_cd - 2.0 * onoa_thk
        pch_major_cd2 = pch_major_cd1 - 2.0 * ponoa_pch_thk_major
        pch_minor_cd2 = pch_minor_cd1 - 2.0 * ponoa_pch_thk_minor

        self.dx = (coordi_x_upper_limit - coordi_x_lower_limit) / coordi_x_elements          # element length
        self.dy = (coordi_y_upper_limit - coordi_y_lower_limit) / coordi_y_elements          # element length

        self.x_nodes = int(coordi_x_elements + 1)        # number of nodes
        self.y_nodes = int(coordi_y_elements + 1)        # number of nodes
        
        x_range = np.linspace(coordi_x_lower_limit, coordi_x_upper_limit, self.x_nodes, dtype=np.float64)
        y_range = np.linspace(coordi_y_lower_limit, coordi_y_upper_limit, self.y_nodes, dtype=np.float64)

        x_range += 1e-9                                                 # avoiding numerical error
        y_range += 1e-9                                                 # avoiding numerical error

        coordi_x, coordi_y = np.meshgrid(x_range, y_range)              # making meshgrid

        # electric potential
        self.phi  = np.ones([self.x_nodes, self.y_nodes], dtype=np.float64)
        self.phi *= (unit_cell_info['bias']['gate'] + unit_cell_info['bias']['pch_e'] + unit_cell_info['bias']['pch_o']) / 3.0

        # gate
        gate_major_cd_half = gate_major_cd / 2.0
        gate_minor_cd_half = gate_minor_cd / 2.0
        
        inside_gate = ( (coordi_x / gate_major_cd_half)**2 + (coordi_y / gate_minor_cd_half)**2 ) <  1.0    # region
        self.bias = (inside_gate == 0.0) * unit_cell_info['bias']['gate']                                   # bias

        # pch
        pch_major_cd1_half = pch_major_cd1 / 2.0
        pch_minor_cd1_half = pch_minor_cd1 / 2.0
        pch1 = ( (coordi_x / pch_major_cd1_half)**2 + (coordi_y / pch_minor_cd1_half)**2 ) <  1.0       # region

        pch_major_cd2_half = pch_major_cd2 / 2.0
        pch_minor_cd2_half = pch_minor_cd2 / 2.0
        pch2 = ( (coordi_x / pch_major_cd2_half)**2 + (coordi_y / pch_minor_cd2_half)**2 ) >  1.0       # region

        ch_cut_cd_half = ch_cut_cd / 2.0
        ch_cut = np.abs(coordi_x) > ch_cut_cd_half      # region

        pch = pch1 & pch2 & ch_cut                                                              # region
        self.bias += ((pch == 1.0) & (coordi_x < 0)) * unit_cell_info['bias']['pch_e']            # bias
        self.bias += ((pch == 1.0) & (coordi_x > 0)) * unit_cell_info['bias']['pch_o']            # bias

        #  
        self.dielectric = inside_gate & (pch == 0)      # region
        self.mat = self.dielectric.copy()               # region

        # debugging
        if False:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self.mat)
            ax[1].imshow(self.bias)
            plt.show()


    def finite_difference_method2(self, overrelaxation, conv_error):
        #
        self.w = overrelaxation
        
        #
        self.x_coeff = self.w * 1.0 / self.dx**2 / ( 2.0 / self.dx**2 + 2.0 / self.dy**2 )
        self.y_coeff = self.w * 1.0 / self.dy**2 / ( 2.0 / self.dx**2 + 2.0 / self.dy**2 )
        self.conv_coeff = 1.0 - self.w

        #
        self.conv_error = conv_error
        
        #
        error = 1.0

        while error > self.conv_error:
            #
            prev_phi = self.phi.copy()
            
            # iterations
            self.phi[1:-1,1:-1] = self.x_coeff * ( self.phi[2:,1:-1] + self.phi[:-2,1:-1] ) + \
                                  self.y_coeff * ( self.phi[1:-1,2:] + self.phi[1:-1,:-2] ) + \
                                  self.conv_coeff * self.phi[1:-1,1:-1]

            # Dirichlet boundary conditions
            self.phi = np.where(self.mat == 0.0, self.bias, self.phi)
            
            # error
            error = np.max( np.abs(prev_phi - self.phi) )

            # debugging
            if self.iteration_no % 1000 == 0:
                output_string = 'iteration = %i, error = %.2e, time = %s' % (self.iteration_no, error, time.ctime())
                print(output_string)

            # iteration no.
            self.iteration_no += 1

        #
        print( 'error = %.2e, iteration = %i ea' % (conv_error, self.iteration_no) )

        #
        print('iteration end time = %s' % time.ctime())

        # electric field
        self.ex = np.gradient(self.phi, axis=0)
        self.ey = np.gradient(self.phi, axis=1)
        self.em = np.sqrt(self.ex*self.ex + self.ey*self.ey)
        

#==============================================
# main
#

# text example
if False:
    geometry_info = {}
    geometry_info['x'] = {'length':11.0, 'elements':110}         # in m, ea
    geometry_info['y'] = {'length':6.0,  'elements':60}         # in m, ea

    bias_conditions = {}
    bias_conditions['x'] = {'left':0.3, 'right':0.7}
    bias_conditions['y'] = {'upper':1.0, 'lower':0.0}

    ex3_5 = ES2D()
    ex3_5.make_rectangular_box(geometry_info)
    ex3_5.dirichlet_boundary_conditions(bias_conditions)
    ex3_5.finite_difference_method(overrelaxation=1.9, conv_error=1e-5)

# silicon work
if True:
    unit_cell_info = {}
    unit_cell_info['coordi_x_limit'] = {'lower':-1000.0, 'upper':1000.0, 'elements': 2000}
    unit_cell_info['coordi_y_limit'] = {'lower':-1000.0, 'upper':1000.0, 'elements': 2000}
    unit_cell_info['ponoa_alo'] = {'thk':[30.0], 'k':9.0}
    unit_cell_info['ponoa_box'] = {'thk':[70.0], 'k':4.5}
    unit_cell_info['ponoa_ctn'] = {'thk':[54.0], 'k':7.5}
    unit_cell_info['ponoa_tox'] = {'thk':[48.0], 'k':5.0}
    unit_cell_info['ponoa_pch'] = {'thk':[70.0, 60.0], 'k':11.7}
    unit_cell_info['gate'] = {'major_CD':1500.0, 'distortion':0.7}
    unit_cell_info['cut_cd'] = 540.0
    unit_cell_info['bias'] = {'gate':7.0, 'pch_e':0.0, 'pch_o':0.0}

    silicon = ES2D()
    silicon.make_unit_cell_geometry(unit_cell_info)
    silicon.finite_difference_method2(overrelaxation=0.2, conv_error=1e-4)

# visualization
if True:
    fig, ax = plt.subplots(2, 2)
    ax[0,0].imshow(silicon.ex)
    ax[0,1].imshow(silicon.ey)
    ax[1,0].imshow(silicon.em)
    ax[1,1].imshow(silicon.phi)
    plt.show()


