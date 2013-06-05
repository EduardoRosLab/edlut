/***************************************************************************
 *                           EDLUTException.cpp                            *
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

#include "../../include/spike/EDLUTException.h"

const char * EDLUTException::Taskmsgs[] ={
	"",
	"Inserting a spike into the heap",
	"Initializing the output",
	"Putting a spike out",
	"Loading the net configuration file",
	"Initializing the spikes heap",
	"Loading the input spikes from file",
	"Scaling neuron tables",
	"Generating the spike prediction time table",
	"Loading neuron table",

	"Loading neuron tables",
	"Loading weights from file",
	"Saving weights to file",
	"Loading the neuron type configuration"
};

const char * EDLUTException::Errormsgs[] ={
	"",
	"Not enough spike heap room",
	"Can't create file of output spikes",
	"Can't write to file of output spikes",
	"Too many state variables defined in the neuron type",
	"Can't allocate enough memory",
	"Can't read the number of neuron types from file",
	"The actual number of neurons doesn't match with the specified total",
	"Can't read enough neuron-type specifications from file",
	"Can't read the number of neurons from file",

	"The number of interconnections doesn't match with the total specified",
	"The neuron specified in interconnections doesn't exist",
	"Can't read enough interconnections from file",
	"Can't read the number of interconnections from file",
	"Can't open the net configuration file",
	"The connection between source and target neuron of the spike has not been defined",
	"Found more input spikes in file that expected",
	"Spike neuron number hasn't been defined",
	"Can't read enough input spikes from the file",
	"Can't read the number of inputs from the file",

	"Can't open the input spikes file",
	"The table doesn't appear to be correct (not enough data)",
	"Can't read enough tables from the file of neuron tables",
	"The table in the file of neuron tablesis empty",
	"Can't open the file of neuron tables of one of the specified neuron types",
	"Can't open the neuron type configuration file",
	"Can't read the number of types of weight change from the file",
	"Weight change constant out of range",
	"Can't read enough types of weight change from file",
	"The type of weight change referenced in the interconnection definition hasn't been defined",

	"Can't open the file of weights",
	"Can't read enough weights from the file",
	"Invalid number of weights value",
	"Can't write to file of weights",
	"Can't read the number of state variables per neuron",
	"Can't read the number of table used for prediction",
	"Can't read the number of synaptic variables per neuron",
	"Can't read the number of tables to load from the file",
	"Can't read the number of dimensions",
	"Can't read the numbers of state variables or interpolation flags corresponding to each dimension",

	"Can't read the numbers of state variables that will be used as synaptic variables",
	"Can't read the numbers of tables used to update the state variables",
	"Can't read the initialization value for each state variable",
	"Can't read the number of table used for end of prediction",
	"Invalid number of neuron types",
	"The table is too big for the current processor/compiler architecture or the table file is corrupt",
	"Can't read the number of the state variable that will be used as last spike time variable",
	"Can't read the number of the state variable that will be used as seed variable",
	"Can't read the time step in the time-driven neuron model",
	"Can't read the relative refractory period",

	"Can't read the absolute refractory period",
	"Can't read the potential gain factor",
	"Can't read the probabilistic threshold potential",
	"Can't read the spontaneous firing rate",
	"Can't read the synaptic efficacy",
	"Can't read the resting potential",
	"Can't read the number of channels in the time-driven neuron model",
	"Can't read the decay time constant parameter",
	"Invalid type of the neuron model",
	"Can't read the gap junction factor",

	"Can't read the resting conductance",
	"Can't read the refractory period",
	"Can't read the gap junction time constant",
	"Can't read the GABA receptor time constant",
	"Can't read the NMDA receptor time constant",
	"Can't read the AMPA receptor time constant",
	"Can't read the membrane capacitance",
	"Can't read the firing threshold",
	"Can't read the resting potential",
	"Can't read the inhibitory reversal potential",
	
	"Can't read the excitatory reversal potential",
	"Can't read the membrane time constant",
	"Can't read the synaptic time constant",
	"Can't read the firing threshold",
	"Can't read the k1 parameter",
	"Can't read the k2 parameter",
	"Can't read the buffer amplitude",
	"Can't read the refractory period"
};

const char * EDLUTException::Repairmsgs[] ={
	"",
	"Allocate more memory for the spike heap",
	"Check for disk problems",
	"Specify these neuron type data in the configuration file or check for errors in the previous lines",
	"Free more memory or use smaller tables",
	"Specify the number of neuron types in the configuration file",
	"Specify correctly the number of neurons in the configuration file",
	"Define more neuron-type specifications in the configuration file",
	"Specify the number of neurons in the configuration file",
	"Specify the correct number of interconnections in the configuration file",

	"Define the neuron or correct the interconnection neuron number",
	"Specify more interconnections or correct the interconnection number in the configuration file",
	"Specify the number of interconnections in the configuration file",
	"Ensure that the file has the proper name and is in the application directory",
	"Define the connection or correct the spike definition in the file of input spikes",
	"Specify the correct number of spikes in the file of input spikes",
	"Specify a correct neuron number in the file of input spikes",
	"Define more spikes or correct the number of spikes in the file of input spikes",
	"Specify the number of spikes in the file of input spikes",
	"Generate a correct file of neuron tables",

	"Specify the correct number of neuron types or change the type of some neurons",
	"Specify the number of types of weight change in the configuration file",
	"Specify correct values for the constants",
	"Specify more types of weight change or correct the number of types of weight change in the configuration file",
	"Define the referenced type of weight change or specify a correct number of type of weight change",
	"Define more weights in the file of weights",
	"Define the exact number of weights of the network",
	"Ensure that the file has the correct permissions to be accessed for reading and writing",
	"Free more memory or use a smaller network",
	"Reduce the number of state variables or change the maximum number of state variables in the simulator source code",

	"Check the type of the neuron model: Only SRMTimeDriven, TableBasedModel and SRMTableBasedModel are implemented at the moment",
	"Check if the neuron model is described and can be accessed by this software"
};

long EDLUTException::GetErrorValue(int a, int b, int c, int d){
	return ((((a)&(long)0xFF)<<24) | (((b)&0xFF)<<16) | (((c)&0xFF)<<8) | ((d)&0xFF));	
}

EDLUTException::EDLUTException(int a, int b, int c, int d){
	this->ErrorNum = GetErrorValue(a,b,c,d);
}

long EDLUTException::GetErrorNum() const{
	return this->ErrorNum;
}
		
const char * EDLUTException::GetTaskMsg() const{
	return Taskmsgs[this->ErrorNum>>24];
}
		
const char * EDLUTException::GetErrorMsg() const{
	return Errormsgs[(this->ErrorNum>>16) & 0xFF];
}
		
const char * EDLUTException::GetRepairMsg() const{
	return Repairmsgs[(this->ErrorNum>>8) & 0xFF];
}		

void EDLUTException::display_error() const{

	if(this->ErrorNum){
		cerr << "Error while: " << this->GetTaskMsg() << endl;
		cerr << "Error message (" << this->ErrorNum << "): " << this->GetErrorMsg() << endl;
		cerr << "Try to: " << this->GetRepairMsg() << endl;
	}
}

ostream & operator<< (ostream & out, EDLUTException Exception){
	if(Exception.GetErrorNum()){
		out << "Error while: " << Exception.GetTaskMsg() << endl;
		if((Exception.GetErrorNum() & 0xFF) == 1){
			//sprintf(msgbuf,"In file line: %li",Currentline);
			//fprintf(stderr,msgbuf);
		}
		
		out << "Error message " << Exception.GetErrorNum() << ": " << Exception.GetErrorMsg() << endl;
		out << "Try to: " << Exception.GetRepairMsg() << endl;
	}
	
	return out;
}

