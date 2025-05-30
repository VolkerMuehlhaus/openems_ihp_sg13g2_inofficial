import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules')))

import modules.util_stackup_reader as stackup_reader
import modules.util_gds_reader as gds_reader
import modules.util_utilities as utilities
import modules.util_simulation_setup as simulation_setup
import modules.util_meshlines as util_meshlines

from openEMS import openEMS
import numpy as np
import matplotlib.pyplot as plt

# Model comments
# 
# Run model with both port excitations, for full S-matrix:
# Get S11 and S21 from port 1 excitation
# Get S22 and S12 from port 2 excitation
# Meshing: simulation_setup.setupSimulation() uses xy_mesh_function=util_meshlines.create_xy_mesh_from_polygons 


# ======================== workflow settings ================================

# preview model/mesh only?
# postprocess existing data without re-running simulation?
preview_only = True
postprocess_only = False

# ===================== input files and path settings =======================

gds_filename = "line_simple.gds"   # geometries
XML_filename = "SG13G2_nosub.xml"               # stackup

# preprocess GDSII for safe handling of cutouts/holes?
preprocess_gds = True

# merge via polygons with distance less than .. microns, set to 0 to disable via merging.
merge_polygon_size = 0

# get path for this simulation file
script_path = utilities.get_script_path(__file__)

# use script filename as model basename
model_basename = utilities.get_basename(__file__)

# set and create directory for simulation output
sim_path = utilities.create_sim_path (script_path, model_basename)
print('Simulation data directory: ', sim_path)

# change current path to model script path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ======================== simulation settings ================================

unit   = 1e-6  # geometry is in microns
margin = 50    # distance in microns from GDSII geometry boundary to simulation boundary 

fstart =  20e9
fstop  = 100e9
numfreq = 401

refined_cellsize = 1.5  # mesh cell size in conductor region

# choices for boundary: 
# 'PEC' : perfect electric conductor (default)
# 'PMC' : perfect magnetic conductor, useful for symmetries
# 'MUR' : simple MUR absorbing boundary conditions
# 'PML_8' : PML absorbing boundary conditions
Boundaries = ['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PEC', 'PML_8']

cells_per_wavelength = 20   # how many mesh cells per wavelength, must be 10 or more
energy_limit = -40          # end criteria for residual energy (dB)

# ports from GDSII Data, polygon geometry from specified special layer
# note that for multiport simulation, excitations are switched on/off in simulation_setup.createSimulation below

simulation_ports = simulation_setup.all_simulation_ports()
# second port has opposite direction, because ground is on the other side
simulation_ports.add_port(simulation_setup.simulation_port(portnumber=1, voltage=1, port_Z0=50, source_layernum=201, target_layername='TopMetal2', direction='x'))
simulation_ports.add_port(simulation_setup.simulation_port(portnumber=2, voltage=1, port_Z0=50, source_layernum=202, target_layername='TopMetal2', direction='-x'))

# ======================== simulation ================================

# get technology stackup data
materials_list, dielectrics_list, metals_list = stackup_reader.read_substrate (XML_filename)
# get list of layers from technology
layernumbers = metals_list.getlayernumbers()
layernumbers.extend(simulation_ports.portlayers)

# read geometries from GDSII, only purpose 0
allpolygons = gds_reader.read_gds(gds_filename, layernumbers, purposelist=[0], metals_list=metals_list, preprocess=preprocess_gds, merge_polygon_size=merge_polygon_size)

# calculate maximum cellsize from wavelength in dielectric
wavelength_air = 3e8/fstop / unit
max_cellsize = (wavelength_air)/(np.sqrt(materials_list.eps_max)*cells_per_wavelength) 



########### create model, run and post-process ###########

FDTD = openEMS(EndCriteria=np.exp(energy_limit/10 * np.log(10)))
FDTD.SetGaussExcite( (fstart+fstop)/2, (fstop-fstart)/2 )
FDTD.SetBoundaryCond( Boundaries )

# Create simulation for port 1 and 2 excitation, return value is list of data paths, one for each excitation
data_paths = []
for excite_ports in [[1],[2]]:  # list of ports that are excited one after another
    FDTD = simulation_setup.setupSimulation (excite_ports, 
                                             simulation_ports, 
                                             FDTD, 
                                             materials_list, 
                                             dielectrics_list, 
                                             metals_list, 
                                             allpolygons, 
                                             max_cellsize, 
                                             refined_cellsize, 
                                             margin, 
                                             unit, 
                                             xy_mesh_function=util_meshlines.create_xy_mesh_from_polygons)
    
    data_paths.append(simulation_setup.runSimulation (excite_ports, FDTD, sim_path, model_basename, preview_only, postprocess_only))


########## evaluation of results with composite GSG ports ###########

if preview_only==False:

    print('Begin data evaluation')

    # define dB function for S-parameters
    def dB(value):
        return 20.0*np.log10(np.abs(value))        

    # define phase function for S-parameters
    def phase(value):
        return np.angle(value, deg=True) 

    f = np.linspace(fstart,fstop,numfreq)


    # get results, CSX port definition is read from simulation ports object
    # S12, S22 is available because we have simulated both port1 and port2  excitation
    s11 = utilities.calculate_Sij (1, 1, f, sim_path, simulation_ports)
    s21 = utilities.calculate_Sij (2, 1, f, sim_path, simulation_ports)
    s12 = utilities.calculate_Sij (1, 2, f, sim_path, simulation_ports)
    s22 = utilities.calculate_Sij (2, 2, f, sim_path, simulation_ports)

    # write Touchstone S2P file
    s2p_name = os.path.join(sim_path, model_basename + '.s2p')
    utilities.write_snp (np.array([[s11, s21],[s12,s22]]),f, s2p_name)

    print('Starting plots')

    fig, axis = plt.subplots(num='Return Loss', tight_layout=True)
    axis.plot(f/1e9, dB(s11), 'k-',  linewidth=2, label='S11 (dB)')
    axis.plot(f/1e9, dB(s22), 'r--', linewidth=2, label='S22 (dB)')
    axis.grid()
    axis.set_xmargin(0)
    axis.set_xlabel('Frequency (GHz)')
    axis.set_title('Return Loss')
    axis.legend()
    
    fig, axis = plt.subplots(num='Insertion Loss', tight_layout=True)
    axis.plot(f/1e9, dB(s21), 'k-',  linewidth=2, label='S21 (dB)')
    axis.plot(f/1e9, dB(s12), 'r--', linewidth=2, label='S12 (dB)')
    axis.grid()
    axis.set_xmargin(0)
    axis.set_xlabel('Frequency (GHz)')
    axis.set_title('Insertion Loss')
    axis.legend()

    fig, axis = plt.subplots(num='Transmission Phase', tight_layout=True)
    axis.plot(f/1e9, phase(s21), 'k-',  linewidth=2, label='S21 (dB)')
    axis.grid()
    axis.set_xmargin(0)
    axis.set_xlabel('Frequency (GHz)')
    axis.set_title('Transmission Phase')
    axis.legend()


   # show all plots
    plt.show()

 


