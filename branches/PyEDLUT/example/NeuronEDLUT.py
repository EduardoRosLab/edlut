#!/usr/bin/python
# coding: UTF-8

# Short demo of co-simulating networks by using Neuron and EDLUT.

import matplotlib.pylab as py
import numpy as np
import random
import neuron
import edlut

def _Create_Neuron_Net_():
    ################################################################################
    ################################################################################
    #                      DEFINE NEURON NETWORK                                   #
    ################################################################################
    ################################################################################
    global golgi
    global AMPA
    global granulestub
    global synapse
    global NewCon
    global potential
    global GolgiCellsOutputTime
    global NeuronTimeVector
    
    # Define the morphological characteristics of each Golgi cell
    golgi = neuron.h.Section()
    golgi.L= 19
    golgi.diam = 19
    golgi.insert("hh")
    golgi.gnabar_hh = 5e-2
    golgi.gkbar_hh = 5e-3
    golgi.gl_hh = 6e-6
    golgi.el_hh = -56
    
    # Define the characteristics of each synapsis to the Golgi cell
    AMPA = neuron.h.ExpSyn(golgi(0.5))
    AMPA.tau = 0.5
    AMPA.e = 0
    
    # Define a stub cell which does not generate activity (only in order to stimulate the Golgi cells)    
    granulestub = neuron.h.NetStim()
    
    # Define the connection between GrC and GoC by using the previousrly defined channel
    synapse = neuron.h.NetCon(granulestub, AMPA)
    #    synapse.threshold = -50 # Not used
    #    synapse.delay = self.delay # Not used
    synapse.weight[0] = GrC_GoC_Weight
        
    # Set Golgi cell spike recording    
    NewCon = neuron.h.NetCon(golgi(0.5)._ref_v, None, sec=golgi)
    NewCon.threshold = -50
    
    # Register Golgi cell membrane potential
    potential = neuron.h.Vector()
    potential.record(golgi(0.5)._ref_v)
    #potential.record(AMPA._ref_g)
    
    # Define Golgi cell records (output spike times)
    GolgiCellsOutputTime = neuron.h.Vector()
    NewCon.record(GolgiCellsOutputTime)
    
    # Define simulation time recording
    NeuronTimeVector = neuron.h.Vector()
    NeuronTimeVector.record(neuron.h._ref_t)
        
    # Initialize neuron simulation
    neuron.h.finitialize()
        
        
        
def _Create_Edlut_Net_():
    ################################################################################
    ################################################################################
    #                      DEFINE EDLUT NETWORK                                    #
    ################################################################################
    ################################################################################

    try:
        edlut.loadnet(network='EDLUTNetNeuron.dat',weights='EDLUTWeightsNeuron.dat')
    except edlut.error as strerror:
        print "EDLUT error({0}): {1}".format(1, strerror)
        raise
    
    print 'Network loaded'


print "Loading file"
neuron.h.load_file("stdrun.hoc")

GrC_GoC_Weight = 5e-4

# Define simulation parameters
simulationtime = 1
communicationtime = 0.002 # Communication time must be below the minimum delay between edlut and neuron networks
delay = 0.002

# Initialize input and output vectors
mossytimes = np.arange(0,simulationtime,0.001).tolist()
granuletimes = []
golgitimes = []


# Create both (Neuron and EDLUT) networks
_Create_Neuron_Net_()
_Create_Edlut_Net_()

# Define input activity for Granule cells
cellfiring = np.zeros(len(mossytimes)).astype(int).tolist()

# Inject excitatory activity to mossy fibers
print 'Injecting spikes'

edlut.injectspikes(spiketimes=mossytimes,firingcells=cellfiring)


StoredGolgiActivity = 0

# Simulation loop
for timestep in np.arange(communicationtime,simulationtime,communicationtime):
    
    # Simulate step between successive communications
    edlut.simulateslot(time=timestep)
    neuron.h.continuerun(timestep*1000)
        
    print 'Simulation time: '+str(timestep)

    # Retrieve EDLUT activity
    outputactivity = edlut.getoutput()
    granulefiringtime = outputactivity[0]
    granuletimes.extend(granulefiringtime)
    
    # Retrieve neuron activity
    Size= GolgiCellsOutputTime.size()
    NewActivityTime = []
    if (Size>StoredGolgiActivity):
        for i in range(StoredGolgiActivity,int(Size)):
            NewActivityTime.append(GolgiCellsOutputTime.x[i])
        StoredGolgiActivity = StoredGolgiActivity + len(NewActivityTime)
        
        print 'Injecting feedback activity to EDLUT'
        
        # Inject input activity to EDLUT
        NewSpikeTimes = []
        NewSpikeIds = (np.ones(len(NewActivityTime))*2).astype(int).tolist()
        for time in NewActivityTime:
            TimeInSec = (time+delay)/1000 # Convert spike time to seconds, adding the connection delay
            NewSpikeTimes.append(TimeInSec)             
        edlut.injectspikes(spiketimes=NewSpikeTimes, firingcells=NewSpikeIds)
    
    print 'Injecting feedback activity to neuron'
    
    # Inject input activity to Neuron
    for i in range(len(granulefiringtime)):
        TimeInMs = (granulefiringtime[i]+delay)*1000
        synapse.event(TimeInMs)
        
# Plot network spikes    
py.figure() #to change the size of the plot use: mp.figure(1, figsize=(20,20))
if (len(mossytimes)>0):
    py.scatter(mossytimes,np.zeros(len(mossytimes)),c='b',marker='o')
py.hold(True)
if (len(granuletimes)>0):
    py.scatter(granuletimes,np.ones(len(granuletimes)),c='r', marker='o')
if (GolgiCellsOutputTime.size()>0):
    TimeFiring = []
    for i in range(int(GolgiCellsOutputTime.size())):
        TimeFiring.append(GolgiCellsOutputTime.x[i]/1000)    
    CellFiring = (np.ones(GolgiCellsOutputTime.size())*2).tolist()
    py.scatter(TimeFiring,CellFiring,c='k', marker='o')
py.xlim(0,0.1)
py.xlabel('Simulation Time (s)')
py.ylim(-0.5,2.5)
py.yticks( np.arange(3), ('MF Input', 'GrC', 'GoC') )
py.draw()

# Plot Golgi cell potentials
py.figure()
py.hold(True)
py.ylabel("Membrane Potential (mV)") 
#py.title(Layer_Names[counter])
# Plot a line per recorded potential vector
py.plot(NeuronTimeVector, potential)
#py.plot(potentialrecording)
py.xlim(0,100)
py.xlabel('Simulation Time (ms)')
py.draw()
py.show()

edlut.finalize()

print 'Network simulation finished'

