/***************************************************************************
 *                           TestSpeed.cpp                                 *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido                        *
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

#include <iostream>

#include "../include/interface/LSAM_Interface.h"

#include "../include/simulation/Simulation.h"

using namespace std;

/** Event-Driven Look-Up Table simulator (EDLUT)

\note Obligatory switches:

\param -time&nbsp;Simulation_Time	 sets the total simulation time (in seconds).
\param -nf&nbsp;Network_File		 sets the network description file.
\param -wf&nbsp;Weights_File		 sets the weights file.

\note Optional switches:

\param -info				 shows network information.
\param -sf&nbsp;File_Name		 saves the final weights in file File_Name.
\param -wt&nbsp;Save_Weight_Step	 sets the step time between weights saving.
\param -st&nbsp;Step_Time(in_seconds)	 sets the step time in simulation.
\param -log&nbsp;File_Name		 saves the activity register in file File_Name.
\param -if&nbsp;Input_File		 adds the Input_File file to the input sources of the simulation.
\param -of&nbsp;Output_File		 adds the Output_File file to the output targets of the simulation.
\param -ic&nbsp;IPDirection:Port&nbsp;Server|Client	 adds the connection as a server or a client at 
                                                         the specified address to the input sources of the simulation.
\param -oc&nbsp;IPDirection:Port&nbsp;Server|Client	 adds the connection as a server or a client at
                                                         the specified address to the output targets of the simulation.	 
\param -ioc&nbsp;IPDirection:Port&nbsp;Server|Client	 adds the connection as a server or a client at the specified
                                                         address to the input sources and in the output targets of the simulation.
\file */

int main(int ac, char *av[]) {
        int result = 0;
	clock_t startt, endt;
	cout << "Loading tables..." << endl;
   
        Simulation * Simul = InitializeAndCatch(ac, av);        // Initialize simulation
        if (Simul == NULL) return 1;                            // Return 1 if there was an error

 	cout << "Simulating network..." << endl;
		
	startt = clock();					// Simulate network and catch errors
        result = SimulateSlotAndCatch(Simul,Simul->GetFinalSimulationTime());
	endt = clock();
         
	cout << "Oky doky" << endl;     

	cout << "Elapsed time: " << (endt-startt)/(float)CLOCKS_PER_SEC << " sec" << endl;
	cout << "Number of updates: " << Simul->GetSimulationUpdates() << endl;
	cout << "Mean number of spikes in heap: " << Simul->GetHeapAcumSize()/(float)Simul->GetSimulationUpdates() << endl;
	cout << "Updates per second: " << Simul->GetSimulationUpdates()/((endt-startt)/(float)CLOCKS_PER_SEC) << endl;
	cout << "Total spikes handled: " << Simul->GetTotalSpikeCounter() << endl;
        return result;
}
