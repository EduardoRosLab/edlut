/***************************************************************************
 *                           StepByStep.cpp                                *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <time.h>
#include <math.h> // TODO: maybe remove this

#include <iostream>

#include "../include/simulation/Simulation.h"
#include "../include/communication/ArrayInputSpikeDriver.h"
#include "../include/communication/ArrayOutputSpikeDriver.h"

#include "../include/spike/EDLUTFileException.h"
#include "../include/spike/EDLUTException.h"

using namespace std;

/*!
 * 
 *
 * \note This software is only an example of how to run step-by-step simulations.
 * 
 * This simulation runs with a 10-ms step and generates 10 random spikes each step 
 * in input cells.
 */ 
int main(int ac, char *av[]) {
   
	int result = 0;
	clock_t startt, endt;
	
	
	const char * NetworkFile = "NetUPMC.dat";
	const char * WeightsFile = "weightsInit.dat";
	double SimulationTime = 1;
	double StepTime = 0.10;

	const int NumberInputCells = 42;

	// Create the new simulation object (and load the network and weight definition file)
	Simulation * Simul = new Simulation(NetworkFile,WeightsFile, SimulationTime);
	
	// Create a new input object to add input spikes
	ArrayInputSpikeDriver * InputDriver = new ArrayInputSpikeDriver();
	Simul->AddInputSpikeDriver(InputDriver);
	
	// Create a new output object to get output spikes
	ArrayOutputSpikeDriver * OutputDriver = new ArrayOutputSpikeDriver();
	Simul->AddOutputSpikeDriver(OutputDriver);
	
	// Get the external initial inputs (none in this simulation)
	Simul->InitSimulation();

	
	startt = clock();					// Simulate network and catch errors
	
	double InputSpikeTimes [NumberInputCells];
	long int InputSpikeCells [NumberInputCells];

	double * OutputSpikeTimes;
	long int * OutputSpikeCells;
		
	// Simulate step by step.
	for (double CurrentTime = 0; CurrentTime<SimulationTime; CurrentTime+=StepTime){
		
		cout << "Simulation at time " << CurrentTime << endl;

		// Generate input spikes (we generate one spike at random time for each input cell)
		for (int i=0; i<NumberInputCells; ++i){
			InputSpikeTimes[i] = rand()*StepTime/RAND_MAX+CurrentTime;
			InputSpikeCells[i] = i;
		}
		
		// Load inputs
		InputDriver->LoadInputs(Simul->GetQueue(),Simul->GetNetwork(),10,InputSpikeTimes,InputSpikeCells);
	
		// Simulate until CurrentTime+StepTime
		Simul->RunSimulationSlot(CurrentTime+StepTime);

		// Get outputs and print them
		int OutputNumber = OutputDriver->GetBufferedSpikes(OutputSpikeTimes,OutputSpikeCells);

		if (OutputNumber>0){
			for (int i=0; i< OutputNumber; ++i){
				cout << "Output spike at time " << OutputSpikeTimes[i] << " from cell " << OutputSpikeCells[i] << endl;
			}

			delete [] OutputSpikeTimes;
			delete [] OutputSpikeCells;
		}
	}
	
	endt = clock();

	cout << "Oky doky" << endl;

	cout << "Elapsed time: " << (endt-startt)/(float)CLOCKS_PER_SEC << " sec" << endl;
	cout << "Number of updates: " << Simul->GetSimulationUpdates() << endl;
	cout << "Mean number of spikes in heap: " << Simul->GetHeapAcumSize()/(float)Simul->GetSimulationUpdates() << endl;
	cout << "Updates per second: " << Simul->GetSimulationUpdates()/((endt-startt)/(float)CLOCKS_PER_SEC) << endl;
	cout << "Total spikes handled: " << Simul->GetTotalSpikeCounter() << endl;

	delete Simul;
	delete InputDriver;
	delete OutputDriver;

	return result;
}
