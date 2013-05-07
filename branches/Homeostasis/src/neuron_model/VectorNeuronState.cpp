/***************************************************************************
 *                           VectorNeuronState.cpp                         *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@atc.ugr.es         *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/VectorNeuronState.h"
#include <string.h>

VectorNeuronState::VectorNeuronState(unsigned int NumVariables, bool isTimeDriven): NumberOfVariables(NumVariables), TimeDriven(isTimeDriven),Is_Monitored(false){
}

VectorNeuronState::VectorNeuronState(const VectorNeuronState & OldState): NumberOfVariables(OldState.NumberOfVariables), SizeStates(OldState.SizeStates), TimeDriven(OldState.TimeDriven),Is_Monitored(OldState.Is_Monitored) {

	VectorNeuronStates = new float[GetNumberOfVariables()*GetSizeState()];
	memcpy(VectorNeuronStates, OldState.VectorNeuronStates, GetNumberOfVariables()*GetSizeState()*sizeof(float));

	LastUpdate=new double[GetSizeState()];
	memcpy(LastUpdate, OldState.LastUpdate, GetSizeState()*sizeof(double));

	LastSpikeTime=new double[GetSizeState()];
	memcpy(LastSpikeTime, OldState.LastSpikeTime, GetSizeState()*sizeof(double));
	
	if(!GetTimeDriven()){
		PredictedSpike=new double[GetSizeState()];
		memcpy(PredictedSpike, OldState.PredictedSpike, GetSizeState()*sizeof(double));

		PredictionEnd=new double[GetSizeState()];
		memcpy(PredictionEnd, OldState.PredictionEnd, GetSizeState()*sizeof(double));
	}
}


VectorNeuronState::VectorNeuronState(const VectorNeuronState & OldState, int index): NumberOfVariables(OldState.NumberOfVariables), SizeStates(1), TimeDriven(OldState.TimeDriven),Is_Monitored(OldState.Is_Monitored) {

	VectorNeuronStates = new float[GetNumberOfVariables()];
	for(int i=0; i<GetNumberOfVariables(); i++){
		VectorNeuronStates[i]=OldState.VectorNeuronStates[index*GetNumberOfVariables()+i];
	}

	LastUpdate=new double[1];
	LastUpdate[0]=OldState.LastUpdate[index];

	LastSpikeTime=new double[1];
	LastSpikeTime[0]=OldState.LastSpikeTime[index];
	
	if(!GetTimeDriven()){
		PredictedSpike=new double[1];
		PredictedSpike[0]=OldState.PredictedSpike[index];

		PredictionEnd=new double[1];
		PredictionEnd[0]=OldState.PredictionEnd[index];
	}
}

VectorNeuronState::~VectorNeuronState() {
	delete [] this->VectorNeuronStates;
	delete [] this->LastUpdate;
	delete [] this->LastSpikeTime;
	if (!TimeDriven){
		delete [] this->PredictedSpike;
		delete [] this->PredictionEnd;
	}
}

void VectorNeuronState::SetStateVariableAt(int index, int position,float NewValue){
	this->VectorNeuronStates[index*NumberOfVariables + position] = NewValue;
}

void VectorNeuronState::SetLastUpdateTime(int index, double NewTime){
	this->LastUpdate[index] = NewTime;
}

void VectorNeuronState::SetNextPredictedSpikeTime(int index, double NextPredictedTime){
	this->PredictedSpike[index] = NextPredictedTime;
}

void VectorNeuronState::SetEndRefractoryPeriod(int index, double NextRefractoryPeriod){
	this->PredictionEnd[index] = NextRefractoryPeriod;
}

unsigned int VectorNeuronState::GetNumberOfVariables(){
	return this->NumberOfVariables;
}

float VectorNeuronState::GetStateVariableAt(int index, int position){
	return VectorNeuronStates[index*NumberOfVariables + position];
}

double VectorNeuronState::GetLastUpdateTime(int index){
	return this->LastUpdate[index];
}

double VectorNeuronState::GetNextPredictedSpikeTime(int index){
	return this->PredictedSpike[index];
}

double VectorNeuronState::GetEndRefractoryPeriod(int index){
	return this->PredictionEnd[index];
}

unsigned int VectorNeuronState::GetNumberOfPrintableValues(){
	return this->GetNumberOfVariables()+3;
}

double VectorNeuronState::GetPrintableValuesAt(int index, int position){
	if (position<this->GetNumberOfVariables()){
		return this->GetStateVariableAt(index, position);
	} else if (position==this->GetNumberOfVariables()) {
		return this->GetLastUpdateTime(index);
	} else if(GetTimeDriven()==true){
		return -1;
	}else if (position==this->GetNumberOfVariables()+1){
		return this->GetNextPredictedSpikeTime(index);
	} else if (position==this->GetNumberOfVariables()+2){
		return this->GetEndRefractoryPeriod(index);
	} else return -1;
}

double VectorNeuronState::GetLastSpikeTime(int index){
	return this->LastSpikeTime[index];
}

void VectorNeuronState::NewFiredSpike(int index){
	this->LastSpikeTime[index] = 0;
}

void VectorNeuronState::AddElapsedTime(int index, double ElapsedTime){

	this->LastSpikeTime[index] += ElapsedTime;
}


void VectorNeuronState::SetSizeState(int size){
	SizeStates=size;
}

int VectorNeuronState::GetSizeState(){
	return SizeStates;
}

void VectorNeuronState::SetTimeDriven(bool isTimeDriven){
	TimeDriven=isTimeDriven;
}

bool VectorNeuronState::GetTimeDriven(){
	return TimeDriven;
}


void VectorNeuronState::InitializeStates(int size, float * initialization){
	SetSizeState(size);
	
	VectorNeuronStates = new float[GetNumberOfVariables()*GetSizeState()]();
	LastUpdate=new double[GetSizeState()]();
	LastSpikeTime=new double[GetSizeState()]();
	
	if(!TimeDriven){
		PredictedSpike=new double[GetSizeState()]();
		PredictionEnd=new double[GetSizeState()]();
	}else{
		InternalSpike=new bool[GetSizeState()]();
	}
	

	for(int z=0; z<GetSizeState()*GetNumberOfVariables(); z+=GetNumberOfVariables()){
		for (int j=0; j<GetNumberOfVariables(); j++){ 
			VectorNeuronStates[z+j]=initialization[j];
		}
	}

	for(int z=0; z<GetSizeState(); z++){
		LastSpikeTime[z]=100.0;
	}

}


bool * VectorNeuronState::getInternalSpike(){
	return InternalSpike;
}


void VectorNeuronState::Set_Is_Monitored(bool monitored){
	Is_Monitored=monitored;
}


bool VectorNeuronState::Get_Is_Monitored(){
	return Is_Monitored;
}
