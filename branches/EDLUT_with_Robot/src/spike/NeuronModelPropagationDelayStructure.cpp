/***************************************************************************
 *                           NeuronModelPropagationDelayStructure.cpp      *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros					   *
 * email                : fnaveros@ugr.es		                           *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/spike/NeuronModelPropagationDelayStructure.h"

#include "../../include/openmp/openmp.h"

#include <cstring>
#include <iostream>

using namespace std;

NeuronModelPropagationDelayStructure::NeuronModelPropagationDelayStructure(){
	AllocatedSize=new int[NumberOfOpenMPQueues]();
	size=new int[NumberOfOpenMPQueues]();
	delays= (double**)new double*[NumberOfOpenMPQueues];
	eventSize = (int **) new int * [NumberOfOpenMPQueues];
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		AllocatedSize[i]=1;
		delays[i]=new double[AllocatedSize[i]];
		eventSize[i]=new int[AllocatedSize[i]];
		FillEventSize(i); 
	}
}

NeuronModelPropagationDelayStructure::~NeuronModelPropagationDelayStructure(){
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		delete delays[i];
		delete eventSize[i];
	}
	delete AllocatedSize;
	delete size;
	delete delays;
	delete eventSize;
}

void NeuronModelPropagationDelayStructure::IncludeNewDelay(int queueIndex, double newDelay){
	int i=0;
	while(i<size[queueIndex] && delays[queueIndex][i]!=newDelay){
		i++;
	}
	if(i==size[queueIndex]){
		if(size[queueIndex]==AllocatedSize[queueIndex]){
			AllocatedSize[queueIndex]*=2;
			double * auxDelays=delays[queueIndex];
			delays[queueIndex]= new double[AllocatedSize[queueIndex]];
			memcpy(delays[queueIndex],auxDelays,sizeof(double)*size[queueIndex]);
			delete auxDelays;

			eventSize[queueIndex]=new int[AllocatedSize[queueIndex]];
			FillEventSize(queueIndex); 
		}

		delays[queueIndex][size[queueIndex]]=newDelay;
		size[queueIndex]++;
		for(int j =size[queueIndex]-1; j>0; j--){
			if(delays[queueIndex][j]<delays[queueIndex][j-1]){
				double aux=delays[queueIndex][j];
				delays[queueIndex][j]=delays[queueIndex][j-1];
				delays[queueIndex][j-1]=aux;
			}
		}
	}
}

int NeuronModelPropagationDelayStructure::GetAllocatedSize(int queueIndex){
	return AllocatedSize[queueIndex];
}

int NeuronModelPropagationDelayStructure::GetSize(int queueIndex){
	return size[queueIndex];
}

double NeuronModelPropagationDelayStructure::GetDelayAt(int queueIndex, int index){
	return delays[queueIndex][index];
}

double * NeuronModelPropagationDelayStructure::GetDelays(int queueIndex){
	return delays[queueIndex];
}


int NeuronModelPropagationDelayStructure::GetEventSize(int queueIndex, int index){
	return eventSize[queueIndex][index];
}

void NeuronModelPropagationDelayStructure::IncrementEventSize(int queueIndex, int index){
	eventSize[queueIndex][index]*=2;
	//cout<<queueIndex<<": "<<eventSize[queueIndex][index]<<endl;
}


void NeuronModelPropagationDelayStructure::FillEventSize(int queueIndex){
	for(int i=0; i<AllocatedSize[queueIndex]; i++){
		eventSize[queueIndex][i]=1;
	}
}
