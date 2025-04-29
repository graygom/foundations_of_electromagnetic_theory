#
# TITLE:
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: foundations of electromagnetic theory, 4th ed., Addision Wesley (1993) 
#


import time
import numpy as np
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
# overrelaxation
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
        x_coeff = self.w / self.dx**2 / ( 2.0 / self.dx**2 + 2.0 / self.dy**2 )
        y_coeff = self.w / self.dy**2 / ( 2.0 / self.dx**2 + 2.0 / self.dy**2 )
        conv_coeff = 1.0 - self.w
        
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
                        self.phi[x_node_cnt,y_node_cnt] = x_coeff * (self.phi[x_node_cnt+1,y_node_cnt] + self.phi[x_node_cnt-1,y_node_cnt]) + \
                                                          y_coeff * (self.phi[x_node_cnt,y_node_cnt+1] + self.phi[x_node_cnt,y_node_cnt-1]) + \
                                                          conv_coeff * self.phi[x_node_cnt,y_node_cnt]
            # error
            error = np.max( np.abs(prev_phi - self.phi) )

            # iteration no.
            self.iteration_no += 1

        #
        print( 'w = %.2f, error = %.2e, iteration = %i ea' % (self.w, conv_error, self.iteration_no) )

        #
        print('iteration end time = %s' % time.ctime())

    

#==============================================
# main
#

geometry_info = {}
geometry_info['x'] = {'length':11.0, 'elements':110}         # in m, ea
geometry_info['y'] = {'length':6.0,  'elements':60}         # in m, ea

bias_conditions = {}
bias_conditions['x'] = {'left':0.3, 'right':0.7}
bias_conditions['y'] = {'upper':1.0, 'lower':0.0}

#
ex3_5 = ES2D()
ex3_5.make_rectangular_box(geometry_info)
ex3_5.dirichlet_boundary_conditions(bias_conditions)
ex3_5.finite_difference_method(overrelaxation=1.9, conv_error=1e-5)

# visualization
if True:
    plt.imshow(ex3_5.phi)
    plt.show()
