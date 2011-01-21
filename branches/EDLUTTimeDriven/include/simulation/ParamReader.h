/***************************************************************************
 *                           ParamReader.h                                 *
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

#ifndef PARAMREADER_H_
#define PARAMREADER_H_

/*!
 * \file ParamReader.h
 *
 * \author Jesus Garrido
 * \date September 2008
 *
 * This file declares a class which read the input parameters and generates the
 * simulation characteristics (inputs, outputs, simulation time, simulation step...)
 */
 
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "./ParameterException.h"
#include "../communication/ConnectionException.h"
#include "../spike/EDLUTException.h"

using namespace std;

class InputSpikeDriver;
class OutputSpikeDriver;
class OutputWeightDriver;
class Simulation;

 
 /*!
 * \class ParamReader
 *
 * \brief Class for read input parameters. 
 *
 * This class abstract methods for get the input parameters in the simulation.
 * 
 * \note Obligatory parameters:
 * 			-time Simulation_Time(in_seconds) It sets the total simulation time.
 * 			-nf Network_File	It sets the network description file.
 * 			-wf Weights_File	It sets the weights file.
 * 			
 * \note  parameters:
 * 			-info 	It shows the network information.
 * 			-log File_Name	It saves the monitored cells activity in file File_Name.
 * 			-sf File_Name	It saves the final weights in file File_Name.
 * 			-wt Save_Weight_Step	It sets the step time between weights saving.
 * 			-st Step_Time(in_seconds) It sets the step time in simulation.
 * 			-if Input_File	It adds the Input_File file in the input sources of the simulation.
 * 			-of Output_File	It adds the Output_File file in the output targets of the simulation.
 * 			-ic IPAddress:Port Server|Client	It adds the connection as a server or a client in the specified direction in the input sources of the simulation.
 * 			-oc IPAddress:Port Server|Client	It adds the connection as a server or a client in the specified direction in the output targets of the simulation.
 * 			-ioc IPAddress:Port Server|Client	It adds the connection as a server or a client in the specified direction in the input sources and in the output targets.	 
 *
 *
 * \author Jesus Garrido
 * \date September 2008
 */
 class ParamReader {
 	
 	private:
 	
 		/*!
 		 * Total simulation time.
 		 */
 		double SimulationTime;
 		
 		/*!
 		 * Network configuration file.
 		 */
 		char * NetworkFile;
 		 
 		/*!
 		 * Weights configuration file.
 		 */
 		char * WeightsFile;
 		
 		/*!
 		 * Save weights step time
 		 */
 		double WeightTime;
 		
 		/*!
 		 * Network info?
 		 */
 		bool NetworkInfo;
 		
 		/*!
 		 * Simulation step time. 
 		 */
 		double SimulationStepTime;

		/*!
 		 * Simulation time-driven step time. 
 		 */
 		double TimeDrivenStepTime;
 		 		
 		/*!
 		 * Input drivers.
 		 */ 	
 		vector<InputSpikeDriver *> InputDrivers;
 		
 		/*!
 		 * Output drivers. 
 		 */ 	
 		vector<OutputSpikeDriver *> OutputDrivers;
 		
 		/*!
 		 * Output monitor. 
 		 */ 	
 		vector<OutputSpikeDriver *> MonitorDrivers;
 		
 		/*!
 		 * Output drivers. 
 		 */ 	
 		vector<OutputWeightDriver *> OutputWeightDrivers;  
 	
 		/*!
 		 * \brief It parses the input arguments.
 		 * 
 		 * It parses the input arguments and gets the simulation configuration.
 		 * 
 		 * \param Number Number of arguments in Arguments.
 		 * \param Arguments Arguments to be parsed. 
 		 * 
 		 * \throw ParameterException When something wrong happens in the parser proccess.
 		 */
 		void ParseArguments(int Number, char ** Arguments) throw (ParameterException, ConnectionException);
 		
 		/*!
 		 * \brief It tests if a file exists.
 		 * 
 		 * It tests if a file exists and if we can read it.
 		 * 
 		 * \param Name The file name.
 		 * 
 		 * \return True if the file exists and we can read it. False in other case.
 		 */
 		bool FileExists(string Name);
 		 
 		
 	public:
 		
 		/*!
 		 * \brief Default constructor.
 		 * 
 		 * It creates a new object with the arguments in the parameters.
 		 * 
 		 * \param ArgNumber Number of arguments in Arg.
 		 * \param Arg Arguments of the simulation.
 		 *
 		 * \throw ParameterException When something wrong happens in the parser proccess.
 		 */
 		ParamReader(int ArgNumber, char ** Arg) throw (ParameterException, ConnectionException);
 		
 		/*!
 		 * \brief It gets the total simulation time.
 		 * 
 		 * It gets the total simulation time. The argument indicator for simulation time
 		 * is -time, so it searchs -time and returns the value as a float.
 		 * 
 		 * \return The total simulation time. -1 if the parameter isn't used.
 		 */
 		double GetSimulationTime();
 		
 		/*!
 		 * \brief It gets the saving weights step time.
 		 * 
 		 * It gets the saving weights step time. The argument indicator for saving weights step time
 		 * is -wt, so it searchs -wt and returns the value as a float.
 		 * 
 		 * \return The saving weights step time. -1 if the parameter isn't used.
 		 */
 		double GetSaveWeightStepTime();
 		
 		/*!
 		 * \brief It gets the network configuration file.
 		 * 
 		 * It gets the network configuration file. The argument indicator for network configuration file
 		 * is -nf, so it searchs -nf and returns the file name.
 		 * 
 		 * \return The network configuration file name. NULL if network file isn't introduced.
 		 */
 		char * GetNetworkFile();
 		
 		/*!
 		 * \brief It gets the weights configuration file.
 		 * 
 		 * It gets the weights configuration file. The argument indicator for weights configuration file
 		 * is -wf, so it searchs -wf and returns the file name.
 		 * 
 		 * \return The weights configuration file name. NULL if weights file isn't introduced.
 		 */
 		char * GetWeightsFile();
 		
 		/*!
 		 * \brief It checks if the -info option is enabled.
 		 * 
 		 * It checks if the -info option is enabled. The argument indicator for network information
 		 * is -info, so it searchs -info.
 		 * 
 		 * \return True if the -info option is enabled. False in other case. 
 		 */ 		
 		bool CheckInfo();
 		
 		/*!
 		 * \brief It gets the simulation step time.
 		 * 
 		 * It gets the simulation step time. The argument indicator for simulation step time
 		 * is -st, so it searchs -st and returns the value as a float.
 		 * 
 		 * \return The simulation step time. -1 if this option isn't enabled. 
 		 */
 		double GetSimulationStepTime();

		/*!
 		 * \brief It gets the simulation time-driven step time.
 		 * 
 		 * It gets the simulation time-driven step time. The argument indicator for simulation step time
 		 * is -ts, so it searchs -ts and returns the value as a float.
 		 * 
 		 * \return The simulation time-driven step time. -1 if this option isn't enabled. 
 		 */
 		double GetTimeDrivenStepTime();
 		
 		
 		/*!
 		 * \brief It gets the input drivers.
 		 * 
 		 * It gets the input drivers of the simulation. The argument indicator for weights configuration file
 		 * is -sf, so it searchs -sf and returns the file name.
 		 * 
 		 * \return A vector of the simulation input drivers. 
 		 */ 	
 		vector<InputSpikeDriver *> GetInputSpikeDrivers();
 		
 		/*!
 		 * \brief It gets the output drivers.
 		 * 
 		 * It gets the output drivers of the simulation. It has two kinds of output drivers:
 		 * -of Output_File, -oc IPAddress:Port Server|Client and -ioc IPAddress:Port Server|Client. It adds all the output drivers.
 		 * 
 		 * \return A vector of the simulation output drivers. 
 		 */ 	
 		vector<OutputSpikeDriver *> GetOutputSpikeDrivers();
 		
 		/*!
 		 * \brief It gets the monitoring drivers.
 		 * 
 		 * It gets the monitoring drivers of the simulation. -log Monitor_File.
 		 * It adds all the monitoring drivers.
 		 * 
 		 * \return A vector of the simulation monitor drivers. 
 		 */ 	
 		vector<OutputSpikeDriver *> GetMonitorDrivers();
 		
 		/*!
 		 * \brief It gets the output weight drivers.
 		 * 
 		 * It gets the output weight drivers of the simulation. The argument indicator for final weights configuration file
 		 * is -sf, so it searchs -sf and returns the file name.
 		 * 
 		 * \return A vector of the simulation output drivers. 
 		 */ 	
 		vector<OutputWeightDriver *> GetOutputWeightDrivers();

 		/*!
		 * \brief It creates and initializes the simulation.
		 *
		 * It creates and initializes the simulation.
		 *
		 * \throw EDLUTException If something wrong happens.
		 * \throw ConnectionException If the connection haven't been able to be stablished.
		 *
		 * \param Reader is the parser of parameters
		 */
		Simulation * CreateAndInitializeSimulation() throw (EDLUTException, ConnectionException);
};

#endif /*PARAMREADER_H_*/
