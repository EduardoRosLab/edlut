/***************************************************************************
 *                           ParamReader.cpp                               *
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

#include "../../include/simulation/ParamReader.h"

#include "../../include/communication/FileInputSpikeDriver.h"
#include "../../include/communication/TCPIPInputSpikeDriver.h"

#include "../../include/communication/FileOutputSpikeDriver.h"
#include "../../include/communication/TCPIPOutputSpikeDriver.h"

#include "../../include/communication/TCPIPInputOutputSpikeDriver.h"

#include "../../include/communication/ServerSocket.h"
#include "../../include/communication/ClientSocket.h"

#include "../../include/communication/FileOutputWeightDriver.h"

#include "../../include/communication/ConnectionException.h"

#include "../../include/simulation/ParameterException.h"
 
void ParamReader::ParseArguments(int Number, char ** Arguments) throw (ParameterException, ConnectionException) {
	for (int i=1; i<Number; ++i){
		string CurrentArgument = Arguments[i];
		if(CurrentArgument=="-time"){ // Simulation Total Time
			if (i+1<Number){
				// Check if it is a number
				istringstream Argument(Arguments[++i]);
   
   				if (!(Argument >> this->SimulationTime))
     				throw ParameterException(Arguments[i], "Invalid simulation time");				
			} else {
				throw ParameterException(Arguments[i],"Invalid simulation time");				
			}
		} else if (CurrentArgument=="-nf"){ // Network configuration file
			if (i+1<Number){
				// Check if it is a valid file and exists
				this->NetworkFile = Arguments[++i];
				if (!this->FileExists(this->NetworkFile)){
					throw ParameterException(Arguments[i],"Invalid network configuration file. The file doesn't exist.");
				}
			} else {
				throw ParameterException(Arguments[i],"Invalid network configuration file");				
			}
		} else if (CurrentArgument=="-wf"){ // Weights configuration file
			if (i+1<Number){
				// Check if it is a valid file and exists
				this->WeightsFile = Arguments[++i];
				if (!this->FileExists(this->WeightsFile)){
					throw ParameterException(Arguments[i],"Invalid weights configuration file. The file doesn't exist.");
				}
			} else {
				throw ParameterException(Arguments[i],"Invalid weights configuration file");				
			}
		} else if (CurrentArgument=="-info"){
			this->NetworkInfo = true;
		} else if (CurrentArgument=="-sf"){
			if (i+1<Number){
				// Check if it is a valid file and exists
				this->OutputWeightDrivers.push_back(new FileOutputWeightDriver (Arguments[++i]));
			}
		} else if (CurrentArgument=="-wt"){
			if (i+1<Number){
				// Check if it is a number
				istringstream Argument(Arguments[++i]);
   
   				if (!(Argument >> this->WeightTime))
     				throw ParameterException(Arguments[i], "Invalid saving weight step time");
			} else {
				throw ParameterException(Arguments[i],"Invalid saving weight step time");				
			}
		} else if (CurrentArgument=="-st"){
			if (i+1<Number){
				// Check if it is a number
				istringstream Argument(Arguments[++i]);
   
   				if (!(Argument >> this->SimulationStepTime))
     				throw ParameterException(Arguments[i], "Invalid simulation step time");
			} else {
				throw ParameterException(Arguments[i],"Invalid simulation step time");				
			}
		} else if (CurrentArgument=="-if"){
			if (i+1<Number){
				// Check if it is a valid file and exists
				string File=Arguments[++i];
				if (!this->FileExists(File)){
					throw ParameterException(File,"Invalid input file. The file doesn't exist.");
				}
				this->InputDrivers.push_back(new FileInputSpikeDriver (File.c_str()));
			} else {
				throw ParameterException(Arguments[i],"Invalid input file");				
			}
		} else if (CurrentArgument=="-ic"){
			if (i+2<Number){
				string host = Arguments[i+1];
				string address = "";
				unsigned short port = 0;
				
				string type = Arguments[i+2];
				
				TCPIPInputSpikeDriver * Driver;
				
				if (type == string("Server")){
					istringstream ss(host);
					if (!(ss >> port)){
	     				throw ParameterException(Arguments[i+1], "Invalid connection port");
					}
					Driver = new TCPIPInputSpikeDriver (new ServerSocket(port));
				}else if (type == string("Client")){
					string::size_type pos = host.find(":",0);
					if ( pos != string::npos ){
						address = host.substr(0, pos);
						istringstream ss(host.substr(pos+1));
						if (!(ss >> port))
	     					throw ParameterException(Arguments[i+1], "Invalid connection port");
					} else {
						// Output error
						throw ParameterException(Arguments[i+1],"Invalid output connection. Check the address and the port.");	
					}
					Driver = new TCPIPInputSpikeDriver (new ClientSocket(address,port));
				} else {
					// Output error
					throw ParameterException(Arguments[i+2],"Invalid output connection type. Only Server and Client are allowed");
				}
				
				i += 2;
				this->InputDrivers.push_back(Driver);
				
			} else {
				throw ParameterException(Arguments[i],"Invalid input connection.");
			}
			
		} else if (CurrentArgument=="-log"){
			if (i+1<Number){
				// Check if it is a valid file and exists
				this->MonitorDrivers.push_back(new FileOutputSpikeDriver (Arguments[++i],false));
			}
		} else if (CurrentArgument=="-logp"){
			if (i+1<Number){
				// Check if it is a valid file and exists
				this->MonitorDrivers.push_back(new FileOutputSpikeDriver (Arguments[++i],true));
			}
		} else if (CurrentArgument=="-of"){
			if (i+1<Number){
				// Check if it is a valid file and exists
				this->OutputDrivers.push_back(new FileOutputSpikeDriver (Arguments[++i],false));
			}
		} else if (CurrentArgument=="-oc"){
			if (i+2<Number){
				string host = Arguments[i+1];
				string address = "";
				unsigned short port = 0;
				
				string type = Arguments[i+2];
				
				TCPIPOutputSpikeDriver * Driver;
				
				if (type == string("Server")){
					istringstream ss(host);
					if (!(ss >> port)){
	     				throw ParameterException(Arguments[i], "Invalid connection port");
					}
					Driver = new TCPIPOutputSpikeDriver (new ServerSocket(port));
				}else if (type == string("Client")){
					string::size_type pos = host.find(":",0);
					if ( pos != string::npos ){
						address = host.substr(0, pos);
						istringstream ss(host.substr(pos+1));
						if (!(ss >> port))
	     					throw ParameterException(Arguments[i], "Invalid connection port");
					} else {
						// Output error
						throw ParameterException(Arguments[i],"Invalid output connection. Check the address and the port.");	
					}
					Driver = new TCPIPOutputSpikeDriver (new ClientSocket(address,port));
				} else {
					// Output error
					throw ParameterException(Arguments[i],"Invalid output connection type. Only Server and Client are allowed");
				}
				
				i += 2;
				
				this->OutputDrivers.push_back(Driver);
			} else {
				throw ParameterException(Arguments[i],"Invalid output connection.");
			}
		} else if (CurrentArgument=="-ioc"){
			if (i+2<Number){
				string host = Arguments[i+1];
				string address = "";
				unsigned short port = 0;
				
				string type = Arguments[i+2];
				
				TCPIPInputOutputSpikeDriver * Driver;
				
				if (type == string("Server")){
					istringstream ss(host);
					if (!(ss >> port)){
	     				throw ParameterException(Arguments[i], "Invalid connection port");
					}
					Driver = new TCPIPInputOutputSpikeDriver (new ServerSocket(port));
				}else if (type == string("Client")){
					string::size_type pos = host.find(":",0);
					if ( pos != string::npos ){
						address = host.substr(0, pos);
						istringstream ss(host.substr(pos+1));
						if (!(ss >> port))
	     					throw ParameterException(Arguments[i], "Invalid connection port");
					} else {
						// Output error
						throw ParameterException(Arguments[i],"Invalid input-output connection. Check the address and the port.");	
					}
					Driver = new TCPIPInputOutputSpikeDriver (new ClientSocket(address,port));
				} else {
					// Output error
					throw ParameterException(Arguments[i],"Invalid input-output connection type. Only Server and Client are allowed");
				}
				
				i += 2;
				
				this->InputDrivers.push_back(Driver);
				this->OutputDrivers.push_back(Driver);
			} else {
				throw ParameterException(Arguments[i],"Invalid input-output connection.");
			}
		} else {
				throw ParameterException(Arguments[i],"Invalid parameter.");
		}	
	}	
	
	if (this->SimulationTime==-1.0){
		throw ParameterException(Arguments[0],"The simulation time isn't specified."); 	
	} else if (this->NetworkFile==NULL){
		throw ParameterException(Arguments[0],"The network configuration file isn't specified.");
	} else if (this->WeightsFile==NULL){
		throw ParameterException(Arguments[0],"The weight configuration file isn't specified.");
	}
	
	return;
}

bool ParamReader::FileExists(string Name){
	bool flag = false;
	fstream fin;
	fin.open(Name.c_str(),ios::in);
	if ( fin.is_open() ){
		flag=true;
	}
	fin.close();
	return flag;	
}
 		 
ParamReader::ParamReader(int ArgNumber, char ** Arg) throw (ParameterException, ConnectionException) :SimulationTime(-1.0), NetworkFile(NULL), WeightsFile(NULL), WeightTime(0.0), NetworkInfo(false),
	SimulationStepTime(0.0), InputDrivers(), OutputDrivers(), OutputWeightDrivers() {
	ParseArguments(ArgNumber,Arg);	
}
 		
double ParamReader::GetSimulationTime(){
	return this->SimulationTime;	
}

double ParamReader::GetSaveWeightStepTime(){
	return this->WeightTime;
}
 		
char * ParamReader::GetNetworkFile(){
	return this->NetworkFile;
}
 		
char * ParamReader::GetWeightsFile(){
	return this->WeightsFile;
}
 		
bool ParamReader::CheckInfo(){
	return this->NetworkInfo;
}
 				
double ParamReader::GetSimulationStepTime(){
	return this->SimulationStepTime;	
}
 		
vector<InputSpikeDriver *> ParamReader::GetInputSpikeDrivers(){
	return this->InputDrivers;
}
 		
vector<OutputSpikeDriver *> ParamReader::GetOutputSpikeDrivers(){
	return this->OutputDrivers;
}

vector<OutputSpikeDriver *> ParamReader::GetMonitorDrivers(){
	return this->MonitorDrivers;
}

vector<OutputWeightDriver *> ParamReader::GetOutputWeightDrivers(){
	return this->OutputWeightDrivers;
}

