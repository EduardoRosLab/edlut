/***************************************************************************
 *                           EDLUTKernel.cpp                               *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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

#include "../include/simulation/ParamReader.h"
#include "../include/simulation/Simulation.h"

#include "../include/simulation/ParameterException.h"

#include "../include/communication/ConnectionException.h"
#include "../include/communication/InputSpikeDriver.h"
#include "../include/communication/OutputSpikeDriver.h"
#include "../include/communication/OutputWeightDriver.h"

#include "../include/spike/EDLUTFileException.h"
#include "../include/spike/EDLUTException.h"
#include "../include/spike/Network.h"




//#include "google/profiler.h"
//#include "vld.h"


using namespace std;

/*!
 * 
 * 
 * \note Obligatory parameters:
 * 			-time Simulation_Time(in_seconds) It sets the total simulation time.
 * 			-nf Network_File	It sets the network description file.
 * 			-wf Weights_File	It sets the weights file.
 * 			
 * \note  parameters:
 * 			-info 	It shows the network information.
 * 			-sf File_Name	It saves the final weights in file File_Name.
 * 			-wt Save_Weight_Step	It sets the step time between weights saving.
 * 			-st Step_Time(in_seconds) It sets the step time in simulation.
 *			-ts Time driven step time (in_seconds).
 *			-tsGPU Time driven step time for GPU (in_seconds).
 * 			-log File_Name It saves the activity register in file File_Name.
 *          -logp File_Name It saves all events register in file File_Name.
 * 			-if Input_File	It adds the Input_File file in the input sources of the simulation.
 * 			-of Output_File	It adds the Output_File file in the output targets of the simulation.
 * 			-ic IPAddress:Port Server|Client	It adds the connection as a server or a client in the specified direction in the input sources of the simulation.
 * 			-oc IPAddress:Port Server|Client	It adds the connection as a server or a client in the specified direction in the output targets of the simulation.
 * 			-ioc IPAddress:Port Server|Client	It adds the connection as a server or a client in the specified direction in the input sources and in the output targets.	 
 * 
  */ 
int main(int ac, char *av[]) {

	clock_t starttotalt,endtotalt;
	starttotalt=clock();

	clock_t startt,endt;
	cout << "Loading tables..." << endl;

	srand ( time(NULL) );

	try {
   		ParamReader Reader(ac, av);
			
		Simulation Simul(Reader.GetNetworkFile(), Reader.GetWeightsFile(), Reader.GetSimulationTime(), Reader.GetSimulationStepTime());
		for (unsigned int i=0; i<Reader.GetInputSpikeDrivers().size(); ++i){
			Simul.AddInputSpikeDriver(Reader.GetInputSpikeDrivers()[i]);
		}
			
		for (unsigned int i=0; i<Reader.GetOutputSpikeDrivers().size(); ++i){
			Simul.AddOutputSpikeDriver(Reader.GetOutputSpikeDrivers()[i]);
		}
		
		for (unsigned int i=0; i<Reader.GetMonitorDrivers().size(); ++i){
			Simul.AddMonitorActivityDriver(Reader.GetMonitorDrivers()[i]);
		}
		
		for (unsigned int i=0; i<Reader.GetOutputWeightDrivers().size(); ++i){
			Simul.AddOutputWeightDriver(Reader.GetOutputWeightDrivers()[i]);
		}
		Simul.SetSaveStep(Reader.GetSaveWeightStepTime());

		if (Reader.GetTimeDrivenStepTime()!=-1){
			Simul.SetTimeDrivenStep(Reader.GetTimeDrivenStepTime());
		}

		if (Reader.GetTimeDrivenStepTimeGPU()!=-1){
			Simul.SetTimeDrivenStepGPU(Reader.GetTimeDrivenStepTimeGPU());
		}
					
		if(Reader.CheckInfo()){
			//Simul.GetNetwork()->tables_info();
			//neutypes_info();
			Simul.PrintInfo(cout);
		}
			
		cout << "Simulating network..." << endl;
		
		startt=clock();
		//ProfilerStart("Profiling.prof");
		Simul.RunSimulation();
		//ProfilerStop();
		endt=clock();

		// Closing simulation connections
		for (unsigned int i=0; i<Reader.GetInputSpikeDrivers().size(); ++i){
			InputSpikeDriver * Input = Reader.GetInputSpikeDrivers()[i];
			delete Input;
		}

		for (unsigned int i=0; i<Reader.GetOutputSpikeDrivers().size(); ++i){
			OutputSpikeDriver * Output = Reader.GetOutputSpikeDrivers()[i];
			delete Output;
		}

		for (unsigned int i=0; i<Reader.GetMonitorDrivers().size(); ++i){
			OutputSpikeDriver * Monitor = Reader.GetMonitorDrivers()[i];
			delete Monitor;
		}

		for (unsigned int i=0; i<Reader.GetOutputWeightDrivers().size(); ++i){
			OutputWeightDriver * Weights = Reader.GetOutputWeightDrivers()[i];
			delete Weights;
		}

		cout << "Oky doky" << endl;  

		cout << "Elapsed time: " << (endt-startt)/(float)CLOCKS_PER_SEC << " sec" << endl;
		cout << "Number of updates: " << Simul.GetSimulationUpdates() << endl;
		cout << "Number of InternalSpike: " << Simul.GetTotalSpikeCounter() << endl;
		cout << "Mean number of spikes in heap: " << Simul.GetHeapAcumSize()/(float)Simul.GetSimulationUpdates() << endl;
		cout << "Updates per second: " << Simul.GetSimulationUpdates()/((endt-startt)/(float)CLOCKS_PER_SEC) << endl;
		
		endtotalt=clock();
		cout << "Total elapsed time: " << (endtotalt-starttotalt)/(float)CLOCKS_PER_SEC << " sec" << endl;



	} catch (ParameterException Exc){
		cerr << Exc << endl;
		cerr << av[0] << " -time Simulation_Time -nf Network_File -wf Weights_File";
		cerr << " [-info] [-sf Final_Weights_File] [-wt Save_Weight_Step] [-st Simulation_Step_Time] [-ts Time_Driven_Step]"; 
		cerr << " [-tsGPU Time_Driven_Step_GPU] [-log Activity_Register_File] [-logp Activity_Register_File] [-if Input_File]";
		cerr << " [-ic IPAddress:Port Server|Client] [-of Output_File] [-oc IPAddress:Port Server|Client] [-ioc IPAddress:Port Server|Client]" << endl;	
	} catch (ConnectionException Exc){
		cerr << Exc << endl;
		return 1;
	} catch (EDLUTFileException Exc){
		cerr << Exc << endl;
		return Exc.GetErrorNum();
	} catch (EDLUTException Exc){
		cerr << Exc << endl;
		return Exc.GetErrorNum();
	}


	return 0;
}
