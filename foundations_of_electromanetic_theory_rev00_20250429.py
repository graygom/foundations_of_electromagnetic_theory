#
# TITLE:
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: foundations of electromagnetic theory, 4th ed., Addision Wesley (1993) 
#


import sys
import time
import pickle
import numpy as np
import scipy as sc
import sympy as sy
import matplotlib.cm as cm
import matplotlib.pyplot as plt


#==============================================
# class ES2D
# REFERENCE = example 3-5. potential in retangular region
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
#
# Poisson equation: ∇·( ε(x,y) * ∇φ(x,y) ) = -ρ(x,y),
# ∇ε(x,y) · ∇φ(x,y) + ε(x,y) * ∇·∇φ(x,y) = -ρ(x,y),
# 
#
#
#
#
#
#
#

class ES2D:

    def __init__(self):
        # geometry
        self.mat = []
        
        # bias conditions
        self.bias = []
        
        # electric potential
        self.phi = []
        
        # electric field
        self.ex = []
        self.ey = []
        self.em = []

        # finite difference method
        self.w = 0.0
        self.conv_error = 0.0

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

        # calculations 1: geometry
        gate_minor_cd = gate_major_cd * gate_distortion
        onoa_thk = np.sum([ponoa_alo_thk, ponoa_box_thk, ponoa_ctn_thk, ponoa_tox_thk]) 
        pch_major_cd1 = gate_major_cd - 2.0 * onoa_thk
        pch_minor_cd1 = gate_minor_cd - 2.0 * onoa_thk
        pch_major_cd2 = pch_major_cd1 - 2.0 * ponoa_pch_thk_major
        pch_minor_cd2 = pch_minor_cd1 - 2.0 * ponoa_pch_thk_minor

        # calculations 2: FDM calculation
        self.dx = (coordi_x_upper_limit - coordi_x_lower_limit) / coordi_x_elements          # element length
        self.dy = (coordi_y_upper_limit - coordi_y_lower_limit) / coordi_y_elements          # element length

        self.x_nodes = int(coordi_x_elements + 1)                                           # number of nodes
        self.y_nodes = int(coordi_y_elements + 1)                                           # number of nodes
        
        self.x_range = np.linspace(coordi_x_lower_limit, coordi_x_upper_limit, self.x_nodes, dtype=np.float64)
        self.y_range = np.linspace(coordi_y_lower_limit, coordi_y_upper_limit, self.y_nodes, dtype=np.float64)

        self.x_range += 1e-9                                                # avoiding numerical error
        self.y_range += 1e-9                                                # avoiding numerical error

        coordi_x, coordi_y = np.meshgrid(self.x_range, self.y_range)        # making meshgrid

        # electric potential (initialization, averaging)
        self.phi  = np.ones([self.x_nodes, self.y_nodes], dtype=np.float64)
        self.phi *= (unit_cell_info['bias']['gate'] + unit_cell_info['bias']['pch_e'] + unit_cell_info['bias']['pch_o']) / 3.0

        # gate region & bias
        gate_major_cd_half = gate_major_cd / 2.0
        gate_minor_cd_half = gate_minor_cd / 2.0
        
        inside_gate = ( (coordi_x / gate_major_cd_half)**2 + (coordi_y / gate_minor_cd_half)**2 ) <  1.0    # region
        self.bias = (inside_gate == 0.0) * unit_cell_info['bias']['gate']                                   # bias

        # pch region & bias
        pch_major_cd1_half = pch_major_cd1 / 2.0
        pch_minor_cd1_half = pch_minor_cd1 / 2.0
        pch1 = ( (coordi_x / pch_major_cd1_half)**2 + (coordi_y / pch_minor_cd1_half)**2 ) <  1.0       # region 1

        pch_major_cd2_half = pch_major_cd2 / 2.0
        pch_minor_cd2_half = pch_minor_cd2 / 2.0
        pch2 = ( (coordi_x / pch_major_cd2_half)**2 + (coordi_y / pch_minor_cd2_half)**2 ) >  1.0       # region 2

        ch_cut_cd_half = ch_cut_cd / 2.0
        ch_cut = np.abs(coordi_x) > ch_cut_cd_half                                                      # region 3

        pch = pch1 & pch2 & ch_cut                                                                      # region 1 & 2 & 3
        
        self.bias += ((pch == 1.0) & (coordi_x < 0)) * unit_cell_info['bias']['pch_e']                  # bias
        self.bias += ((pch == 1.0) & (coordi_x > 0)) * unit_cell_info['bias']['pch_o']                  # bias

        # FDM calculation region
        self.dielectric = inside_gate & (pch == 0)      # region = inside gate & outside channel
        self.mat = self.dielectric.copy()               # region where FDM calculations are run

        # debugging
        if False:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self.mat)
            ax[1].imshow(self.bias)
            plt.show()


    def before_finite_difference_method(self, overrelaxation, conv_error):
        #
        self.w = overrelaxation
        
        #
        self.x_coeff = self.w * 1.0 / self.dx**2 / ( 2.0 / self.dx**2 + 2.0 / self.dy**2 )
        self.y_coeff = self.w * 1.0 / self.dy**2 / ( 2.0 / self.dx**2 + 2.0 / self.dy**2 )
        self.conv_coeff = 1.0 - self.w

        #
        self.conv_error = conv_error


    def finite_difference_method2(self):
        #
        print('FDM iteration strat time = %s' % time.ctime())
        
        #
        start_time = time.time()
        error = 1.0

        while error > self.conv_error:
            # error check reference
            prev_phi = self.phi.copy()
            
            # FDM iterations using numpy array operations
            self.phi[1:-1,1:-1] = self.x_coeff * ( self.phi[2:,1:-1] + self.phi[:-2,1:-1] ) + \
                                  self.y_coeff * ( self.phi[1:-1,2:] + self.phi[1:-1,:-2] ) + \
                                  self.conv_coeff * self.phi[1:-1,1:-1]

            # Dirichlet boundary conditions
            self.phi = np.where(self.mat == 0.0, self.bias, self.phi)
            
            # calculating error
            error = np.max( np.abs(prev_phi - self.phi) )

            # debugging
            if self.iteration_no % 1000 == 0:
                elapsed_time = time.time() - start_time
                output_string = 'iteration = %i, error = %.2e, elapsed time = %.1fmin' % (self.iteration_no, error, elapsed_time/60.0)
                print(output_string)

            # calculating iteration numnber
            self.iteration_no += 1

        #
        print( 'target error = %.2e, iterations = %i ea' % (self.conv_error, self.iteration_no) )

        #
        print('FDM iteration end time = %s' % time.ctime())

        # calculating electric field
        self.ex = np.gradient(self.phi, axis=0)
        self.ey = np.gradient(self.phi, axis=1)
        self.em = np.sqrt(self.ex*self.ex + self.ey*self.ey)

        # Dirichlet boundary conditions (post-processing metal region)
        self.ex = np.where(self.mat == 0.0, 0.0, self.ex)
        self.ey = np.where(self.mat == 0.0, 0.0, self.ey)
        self.em = np.where(self.mat == 0.0, 0.0, self.em)


    def finite_difference_method_error_check(self):
        # error check reference
        prev_phi = self.phi.copy()
            
        # FDM iteration 1ea
        for x_node_cnt in range(self.x_nodes):
            for y_node_cnt in range(self.y_nodes):
                # material check: dielectrics
                if self.mat[x_node_cnt,y_node_cnt] != 0:
                    self.phi[x_node_cnt,y_node_cnt] = self.x_coeff * (self.phi[x_node_cnt+1,y_node_cnt] + self.phi[x_node_cnt-1,y_node_cnt]) + \
                                                      self.y_coeff * (self.phi[x_node_cnt,y_node_cnt+1] + self.phi[x_node_cnt,y_node_cnt-1]) + \
                                                      self.conv_coeff * self.phi[x_node_cnt,y_node_cnt]

        # calculating error
        error = np.max( np.abs(prev_phi - self.phi) )
        print( 'finite_difference_method_error_check(), checked error = %.2e' % error )


    def save_FDM_result(self, output_filename):
        #
        fid = open(output_filename, 'wb')

        #
        pickle.dump([self.mat, self.bias, self.phi, self.ex, self.ey, self.em, self.w, self.conv_error, self.iteration_no], fid)

        #
        fid.close()


    def load_FDM_result(self, input_filename):
        #
        fid = open(input_filename, 'rb')

        #
        self.mat, self.bias, self.phi, self.ex, self.ey, self.em, self.w, self.conv_error, self.iteration_no = pickle.load(fid)

        #
        fid.close()


    def analysis(self):
        #
        x_center_index = int( (self.x_nodes - 1) / 2.0 )
        y_center_index = int( (self.y_nodes - 1) / 2.0 )

        #
        phi_at_x_center = self.phi[x_center_index, :]
        phi_at_y_center = self.phi[:, y_center_index]

        #
        em_at_x_center = self.em[x_center_index, :]
        em_at_y_center = self.em[:, y_center_index]

        #
        coordi_x, coordi_y = np.meshgrid(self.x_range, self.y_range) 

        # text information
        fdm_nodes = np.sum(np.where(self.mat != 0.0, 1.0, 0.0))
        dirichlet_nodes = np.sum(np.where(self.mat == 0.0, 1.0, 0.0))
        print('Number of FDM nodes = %i ea' % fdm_nodes)
        print('Number of Dirichlet nodes = %i ea' % dirichlet_nodes)

        # visualization 1
        fig, ax = plt.subplots(1, 1)
        #
        plt.imshow(self.mat, cmap=cm.jet)
        plt.axis('equal')
        plt.grid(ls=':')
        plt.colorbar()
        plt.title('Materials')
        plt.savefig('figure_1_mat.pdf')

        # visualization 2
        fig, ax = plt.subplots(1, 1)
        #
        plt.imshow(self.bias, cmap=cm.jet)
        plt.axis('equal')
        plt.grid(ls=':')
        plt.colorbar()
        plt.title('Bias')
        plt.savefig('figure_2_bias.pdf')

        # visualization 3
        fig, ax = plt.subplots(1, 1)
        #
        CS1 = plt.contour(coordi_x, coordi_y, self.phi, levels=15, cmap=cm.jet)
        plt.axis('equal')
        plt.grid(ls=':')
        plt.colorbar()
        plt.title('Electric Potential')
        plt.savefig('figure_3_phi.pdf')

        # visualization 4
        fig, ax = plt.subplots(1, 1)
        #
        CS2 = plt.contour(coordi_x, coordi_y, self.em, levels=30, cmap=cm.jet)
        plt.axis('equal')
        plt.grid(ls=':')
        plt.colorbar()
        plt.title('Electric Field Magnitude')
        plt.savefig('figure_4_em.pdf')

        # visualization 5
        fig, ax = plt.subplots(1, 1)
        #
        CSF1 = plt.contourf(coordi_x, coordi_y, self.phi, levels=10, cmap=cm.jet)   # cmap="RdBu_r"
        plt.axis('equal')
        plt.grid(ls=':')
        plt.colorbar()
        plt.title('Electric Potential')
        plt.savefig('figure_5_phi_f.pdf')

        # visualization 6
        fig, ax = plt.subplots(1, 1)
        #
        CSF2 = plt.contourf(coordi_x, coordi_y, self.em, levels=100, cmap=cm.jet)   # cmap="RdBu_r"
        plt.axis('equal')
        plt.grid(ls=':')
        plt.colorbar()
        plt.title('Electric Field Magnitude')
        plt.savefig('figure_6_em_f.pdf')

        plt.close()


    def analysis2(self, unit_cell_info):
        #
        onoa_alo_thk = unit_cell_info['ponoa_alo']['thk'][0]
        onoa_box_thk = unit_cell_info['ponoa_box']['thk'][0]
        onoa_ctn_thk = unit_cell_info['ponoa_ctn']['thk'][0]
        onoa_tox_thk = unit_cell_info['ponoa_tox']['thk'][0]

        onoa_thk = np.sum([onoa_alo_thk, onoa_box_thk, onoa_ctn_thk, onoa_tox_thk])
        
        #
        x_center_index = int( (self.x_nodes - 1) / 2.0 )
        y_center_index = int( (self.y_nodes - 1) / 2.0 )

        #
        phi_at_x_center = self.phi[x_center_index, :]
        phi_at_y_center = self.phi[:, y_center_index]

        #
        em_at_x_center = self.em[x_center_index, :]
        em_at_y_center = self.em[:, y_center_index]

        #
        coordi_x, coordi_y = np.meshgrid(self.x_range, self.y_range) 

        # text information
        fdm_nodes = np.sum(np.where(self.mat != 0.0, 1.0, 0.0))
        dirichlet_nodes = np.sum(np.where(self.mat == 0.0, 1.0, 0.0))
        print('Number of FDM nodes = %i ea' % fdm_nodes)
        print('Number of Dirichlet nodes = %i ea' % dirichlet_nodes)
        
        # Sympy
        major_r = unit_cell_info['gate']['major_CD'] / 2.0
        minor_r = major_r *unit_cell_info['gate']['distortion']

        major_r2 = major_r - onoa_thk
        minor_r2 = minor_r - onoa_thk

        # Sympy
        sy_theta = sy.symbols('theta')
        sy_x3 = sy.symbols('x')
        
        # Sympy
        sy_x = major_r * sy.cos(sy_theta)
        sy_y = minor_r * sy.sin(sy_theta)

        sy_x2 = major_r2 * sy.cos(sy_theta)
        sy_y2 = minor_r2 * sy.sin(sy_theta)

        dsy_x2_dtheta = sy_x2.diff(sy_theta)
        dsy_y2_dtheta = sy_y2.diff(sy_theta)

        sy_normal = -dsy_x2_dtheta / dsy_y2_dtheta
        
        #
        sy_x_eval = []
        sy_y_eval = []

        sy_x_eval2 = []
        sy_y_eval2 = []

        sy_x_eval3 = []
        sy_y_eval3 = []

        #
        for theta in np.linspace(0.0, 2*np.pi-2*np.pi/50, 50):
            sy_x_eval.append(sy_x.subs(sy_theta, theta))
            sy_y_eval.append(sy_y.subs(sy_theta, theta))

            sy_x_eval2.append(sy_x2.subs(sy_theta, theta))
            sy_y_eval2.append(sy_y2.subs(sy_theta, theta))

            #
            sy_x_eval3.append([])
            sy_y_eval3.append([])
            
            sy_normal_line_x = sy_x2 + sy_x3
            sy_normal_line_y = sy_y2 + (sy_normal) * sy_x3

            normal = sy_normal.subs(sy_theta, theta)

            for x in [10, 20, 30]:
                sy_x_eval3[-1].append(sy_normal_line_x.subs( [(sy_theta, theta), (sy_x3, x)] ))
                sy_y_eval3[-1].append(sy_normal_line_y.subs( [(sy_theta, theta), (sy_x3, x)] ))

            #print(sy_x_eval2[-1], sy_y_eval2[-1], normal, sy_x_eval3[-1], sy_y_eval3[-1])

        # visualization 3
        fig, ax = plt.subplots(1, 1)
        #
        CS1 = plt.contour(coordi_x, coordi_y, self.phi, levels=15, cmap=cm.jet)
        plt.axis('equal')
        plt.grid(ls=':')
        plt.colorbar()
        plt.title('Electric Potential')
        plt.savefig('figure_3_phi_.pdf')

        #
        plt.plot(sy_x_eval, sy_y_eval, '.')
        plt.plot(sy_x_eval2, sy_y_eval2, '.')
        for cnt in range(len(sy_x_eval3)):
            plt.plot(sy_x_eval3[cnt], sy_y_eval3[cnt], 'o-')
        plt.axis('equal')
        plt.show()
        

        

#==============================================
# main
#

geometry_info = {}
geometry_info['x'] = {'length':11.0, 'elements':110}            # in m, ea
geometry_info['y'] = {'length':6.0,  'elements':60}             # in m, ea

bias_conditions = {}
bias_conditions['x'] = {'left':0.3, 'right':0.7}
bias_conditions['y'] = {'upper':1.0, 'lower':0.0}

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

# text example
if False:
    ex3_5 = ES2D()
    ex3_5.make_rectangular_box(geometry_info)
    ex3_5.dirichlet_boundary_conditions(bias_conditions)
    ex3_5.finite_difference_method(overrelaxation=1.9, conv_error=1e-5)

# silicon work
if True:
    silicon = ES2D()
    silicon.make_unit_cell_geometry(unit_cell_info)
    
    if False:
        silicon.before_finite_difference_method(overrelaxation=0.2, conv_error=1e-5)
        silicon.finite_difference_method2()
        silicon.finite_difference_method_error_check()
        silicon.save_FDM_result(output_filename='FDM_result_150nm_0.7_cut_54nm_onoa_30_70_54_46_pch_70_60.pkl')

    if False:
        silicon.load_FDM_result(input_filename='FDM_result_150nm_0.7_cut_54nm_onoa_30_70_54_46_pch_70_60_8e-6.pkl')
        silicon.before_finite_difference_method(overrelaxation=0.2, conv_error=5e-6)
        silicon.finite_difference_method2()
        silicon.finite_difference_method_error_check()
        silicon.save_FDM_result(output_filename='FDM_result_150nm_0.7_cut_54nm_onoa_30_70_54_46_pch_70_60_5e-6.pkl')
        silicon.analysis()

    if False:
        silicon.load_FDM_result(input_filename='FDM_result_150nm_0.7_cut_54nm_onoa_30_70_54_46_pch_70_60_5e-6.pkl')
        silicon.before_finite_difference_method(overrelaxation=0.2, conv_error=1e-6)
        silicon.finite_difference_method2()
        silicon.finite_difference_method_error_check()
        silicon.save_FDM_result(output_filename='FDM_result_150nm_0.7_cut_54nm_onoa_30_70_54_46_pch_70_60_1e-6.pkl')
        silicon.analysis()

    if True:
        silicon.load_FDM_result(input_filename='FDM_result_150nm_0.7_cut_54nm_onoa_30_70_54_46_pch_70_60_5e-6.pkl')
        #silicon.analysis()
        silicon.analysis2(unit_cell_info)

# visualization
if False:
    fig, ax = plt.subplots(2, 2)
    ax[0,0].imshow(silicon.ex)
    ax[0,1].imshow(silicon.ey)
    ax[1,0].imshow(silicon.em)
    ax[1,1].imshow(silicon.phi)
    plt.show()





