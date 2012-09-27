#!/usr/bin/env python

import edlut as edlut
import random
import matplotlib.pylab as py
import numpy as np

# Simulation variables
simulationtime = 0.3
communicationtime = 0.002
timedriventimestep = 1e-4

edlut.loadnet('EDLUTNet100.dat','EDLUTWeights100.dat',timedriventimestep)

print 'Network loaded'

# Input variables
inputfrequency = 200
inputcells = 100

inputactivitycell = []
inputactivitytime = []
outputactivitycell = []
outputactivitytime = []

for timestep in np.arange(communicationtime,simulationtime,communicationtime):
    cellfiring = []
    timefiring = []
    for cell in range(0,inputcells,1):
        if (random.random()<communicationtime*inputfrequency):
            cellfiring.append(cell)
            spiketime = timestep-random.random()*communicationtime
            timefiring.append(spiketime)
    
    inputactivitycell.extend(cellfiring)
    inputactivitytime.extend(timefiring)
    
    print 'Injecting spikes'
    
    edlut.injectspikes(timefiring,cellfiring)
        
    print 'Activity injected'
    
    edlut.simulateslot(timestep)
    print 'Simulation time: '+str(timestep)

    outputactivity = edlut.getoutput()
    outputactivitytime.extend(outputactivity[0])
    outputactivitycell.extend(outputactivity[1])
    
    print 'Activity extracted'
    
py.figure()
py.scatter(outputactivitytime,outputactivitycell,c='b',marker='o')
py.hold(True)
py.scatter(inputactivitytime,inputactivitycell,c='r', marker='o')
py.xlim(0,simulationtime)
py.ylim(0,max(outputactivitycell)+1)
py.draw()
py.show()

edlut.finalize()

print 'Network destroyed'
