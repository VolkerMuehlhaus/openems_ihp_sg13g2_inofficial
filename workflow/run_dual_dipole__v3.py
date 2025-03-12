import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules')))

import modules.util_stackup_reader as stackup_reader
import modules.util_gds_reader as gds_reader
import modules.util_utilities as utilities
import modules.util_simulation_setup as simulation_setup
import modules.util_meshlines as util_meshlines

# import pylab as pl # cf. https://matplotlib.org/stable/api/pylab.html
from CSXCAD import ContinuousStructure
from CSXCAD import AppCSXCAD_BIN
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0, MUE0, Z0 # with * unwanted components (variables, instances, etc.) might be imported and can mess up the name space. Here, the module numpy (as np) is imported, which is confusing, as it is no physical constant.
import numpy as np
import matplotlib.pyplot as plt

# Model comments
#
# Dual dipole 245 GHz, design by IHP Klaus Schmalz
# Port defined in GDSIIfile on layer 201
# nf2ff_box field sampling at boundary for pattern calculation


# ======================== workflow settings ================================

# preview model/mesh only?
# postprocess existing data without re-running simulation?
preview_only = False
postprocess_only = False

# ===================== input files and path settings =======================

gds_filename = "dipole_port_sg13.gds"   # geometries
XML_filename = "SG13G2_200um.xml"       # stackup

# preprocess GDSII for safe handling of cutouts/holes?
preprocess_gds = True

# merge via polygons with distance less than .. microns, set to 0 to disable via merging.
merge_polygon_size = 0

# get path for this simulation file
script_path = utilities.get_script_path(__file__)

# use script filename as model basename
model_basename = utilities.get_basename(__file__)

# set and create directory for simulation output
sim_path = utilities.create_sim_path (script_path,model_basename)
print('Simulation data directory: ', sim_path)


# ======================== simulation settings ================================

unit_m = 1e-6   # geometry dimensions and all lengths unit is µm (micrometer)
margin = 100    # distance in microns from GDSII geometry boundary to chip boundary 

f_start = 200e9
f_stop  = 300e9
num_freq = 401

f_target = 245e9  # frequency for antenna pattern calculation

refined_cellsize = 2.5  # mesh cell size in conductor region

# choices for boundary: 
# 'PEC' : perfect electric conductor (default)
# 'PMC' : perfect magnetic conductor, useful for symmetries
# 'MUR' : simple MUR absorbing boundary conditions
# 'PML_8' : PML absorbing boundary conditions
Boundaries = ['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8']

cells_per_wavelength = 12   # how many mesh cells per wavelength, must be 10 or more
energy_limit = -40          # end criteria for residual energy (dB)

# ports from GDSII Data, polygon geometry from specified special layer
# note that for multiport simulation, excitations are switched on/off in simulation_setup.createSimulation below
simulation_ports = simulation_setup.all_simulation_ports()
simulation_ports.add_port(simulation_setup.simulation_port(portnumber=1, voltage=1, port_Z0=50, source_layernum=201, target_layername='TopMetal2', direction='x'))

# ======================== simulation ================================

# get technology stackup data
materials_list, dielectrics_list, metals_list = stackup_reader.read_substrate (XML_filename)
# get list of layers from technology
layernumbers = metals_list.getlayernumbers()
layernumbers.extend(simulation_ports.portlayers)

# read geometries from GDSII, only purpose 0
allpolygons = gds_reader.read_gds(gds_filename, layernumbers, purposelist=[0], metals_list=metals_list, preprocess=preprocess_gds, merge_polygon_size=merge_polygon_size)

# calculate maximum cellsize from wavelength in dielectric
wavelength_air = 3e8/f_stop / unit_m
max_cellsize = (wavelength_air)/(np.sqrt(materials_list.eps_max)*cells_per_wavelength) 


########### create model, run and post-process ###########

# Prepare simulation for port 1 excitation
excite_ports = [1]  # list of ports that are excited for this simulation run
FDTD = openEMS(EndCriteria=np.exp(energy_limit/10 * np.log(10)))
FDTD.SetGaussExcite( (f_start+f_stop)/2, (f_stop-f_start)/2 )
FDTD.SetBoundaryCond( Boundaries )

FDTD = simulation_setup.setupSimulation(
    excite_ports, 
    simulation_ports, 
    FDTD, 
    materials_list, 
    dielectrics_list, 
    metals_list, 
    allpolygons, 
    max_cellsize, 
    refined_cellsize, 
    margin, 
    unit_m, 
    xy_mesh_function = util_meshlines.create_xy_mesh_from_polygons, 
    air_around = 0.5*wavelength_air
    )

# add nf2ff box for antenna pattern calculation
nf2ff_box = FDTD.CreateNF2FFBox(opt_resolution = [max_cellsize]*3, frequency = [f_target])

# run simulation
sub1_data_path = simulation_setup.runSimulation(excite_ports, FDTD, sim_path, model_basename, preview_only, postprocess_only)

# simulation is finished, get results, CSX port definition is read from simulation ports object
CSX_port1 = simulation_ports.get_port_by_number(1).CSXport

# definition of some utility functions
def dB(value, factor=20):
    return factor*np.log10(np.abs(value))
def dBm(value, factor=20):
    return dB(value/1e-3, factor=factor)

# evaluate results for 1-port simulation
if not preview_only:  
    f = np.linspace(f_start, f_stop, num_freq)

    s11 = utilities.calculate_Sij(1, 1, f, sim_path, simulation_ports)
    s11_dB = dB(s11)

    # write Touchstone S1P file
    s1p_name = os.path.join(sim_path, model_basename + '.s1p')
    utilities.write_snp (np.array([s11]),f, s1p_name)

    # plot return loss
    fig, axis = plt.subplots(num="Return Loss", tight_layout=True)
    axis.plot(f/1e9, s11_dB, 'k-', linewidth=2, label='S11 (dB)')
    axis.grid()
    axis.set_xmargin(0)
    axis.set_xlabel('Frequency (GHz)')
    axis.set_title("Return Loss")
    axis.legend()
    
    print('Calculating antenna pattern, this will take a while!')
    
    theta = np.arange(-180.0, 180.0, 2.0)
    phi   = [0., 90.]
    nf2ff_res = nf2ff_box.CalcNF2FF(sub1_data_path, f_target, theta, phi)

    # INPUT POWER values into feed port at evaluation frequency
    # P_acc = accepted power (incoming - reflected)
    # P_inc = incident power (incoming), regardless of reflection
    
    # get accepted power into port
    Pin_accepted = np.interp(f_target, f, CSX_port1.P_acc)
    # get incident power
    P_incident = np.interp(f_target, f, CSX_port1.P_inc)

    # get total radiated power at nf2ff frequency 
    Prad_total = nf2ff_res.Prad[0]
    # get directivity (peak value) at nf2ff frequency 
    Dmax_dB = dB(nf2ff_res.Dmax[0], 10)
    # calculate radiation efficiency (lossless antenna is 1, regardless of input matching)
    radiation_efficiency = Prad_total/Pin_accepted
    radiation_efficiency_dB = dB(radiation_efficiency, 10)  # negative for lossy antenna
    # loss from reflection due to mismatch
    mismatch_loss =  Pin_accepted/P_incident
    mismatch_loss_dB = dB(mismatch_loss, 10)

    # evaluate far field components
    # normalize E field components to peak value of absolute field, and multiply by directivity
    directivity_abs  = dB(nf2ff_res.E_norm[0]/np.max(nf2ff_res.E_norm[0])) + Dmax_dB
    directivity_abs_xz = directivity_abs[:,0] # get field in xz plane, that is phi=0, which is index 0 
    directivity_abs_yz = directivity_abs[:,1] # get field in yz plane, that is phi=90, which is index 1 

    gain_abs_xz = directivity_abs_xz + radiation_efficiency_dB
    gain_abs_yz = directivity_abs_yz + radiation_efficiency_dB

    # display radiated power and directivity
    print(f"Antenna parameters at {f_target/1e9:g} GHz:")
    print(f"    Incident power P_incident  =  {P_incident:.3e} W  =  {dBm(P_incident, 10):g} dBm")
    print(f"    Accepted power Pin_accepted  =  {Pin_accepted:.3e} W  =  {dBm(Pin_accepted, 10):g} dBm")
    print(f"    Radiated power Prad_total  =  {Prad_total:.3e} W  =  {dBm(Prad_total, 10):g} dBm")
    print(f"    Directivity Dmax  =  {Dmax_dB:g} dBi")
    print(f"    efficiency nu_rad  =  {100*radiation_efficiency:g} %  =  {radiation_efficiency_dB:g} dB")
    print(f"    mismatch loss  =  {mismatch_loss:g} (linear)  =  {mismatch_loss_dB:g} dB")


    fig, axis = plt.subplots(num="Directivity", tight_layout=True)
    axis.plot(theta, np.squeeze(directivity_abs_xz), 'k-',  linewidth=2, label='xz-plane')
    axis.plot(theta, np.squeeze(directivity_abs_yz), 'r--', linewidth=2, label='yz-plane')
    axis.grid()
    axis.set_xmargin(0)
    axis.set_xlabel('Theta (deg)')
    axis.set_ylabel('Directivity (dBi)')
    axis.set_title(f'Directivity at Frequency: {f_target/1e9:g} GHz')
    axis.legend()

    fig, axis = plt.subplots(num="Gain", tight_layout=True)
    axis.plot(theta, np.squeeze(gain_abs_xz), 'k-',  linewidth=2, label='xz-plane')
    axis.plot(theta, np.squeeze(gain_abs_yz), 'r--', linewidth=2, label='yz-plane')
    axis.grid()
    axis.set_xmargin(0)
    axis.set_xlabel('Theta (deg)')
    axis.set_ylabel('Gain (dBi)')
    axis.set_title(f'Gain at Frequency: {f_target/1e9:g} GHz')
    axis.legend()

    # show all plots
    plt.show()
