/***************************************************************************
 *                           NeuronPropagationDelayStructure.cpp           *
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

#include "../../include/spike/NeuronPropagationDelayStructure.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/NeuronModelPropagationDelayStructure.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/openmp/openmp.h"
#include <iostream>
using namespace std;

NeuronPropagationDelayStructure::NeuronPropagationDelayStructure(Neuron * neuron){
	NDifferentDelays=new int[NumberOfOpenMPQueues]();
	NSynapsesWithEqualDelay=(int **) new int *[NumberOfOpenMPQueues];
	SynapseDelay=(double **) new double *[NumberOfOpenMPQueues];
	OutputConnectionsWithEquealDealy=(Interconnection***)new Interconnection**[NumberOfOpenMPQueues];
	IndexSynapseDelay=(int **) new int *[NumberOfOpenMPQueues];

	double delay1, delay2;
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		if (neuron->GetOutputNumber(i)!=0){
			NDifferentDelays[i]++;
			delay1=neuron->GetOutputConnectionAt(i,0)->GetDelay();
			for(int j=1; j<neuron->GetOutputNumber(i); j++){
				delay2=neuron->GetOutputConnectionAt(i,j)->GetDelay();
				if(delay1!=delay2){
					delay1=delay2;
					NDifferentDelays[i]++;
				}
			}
		}
	}

	for(int i=0; i<NumberOfOpenMPQueues; i++){
		if(NDifferentDelays[i]!=0){
			NSynapsesWithEqualDelay[i]=(int *) new int [NDifferentDelays[i]]();
			SynapseDelay[i]=(double *) new double [NDifferentDelays[i]]();
			OutputConnectionsWithEquealDealy[i]=(Interconnection**)new Interconnection*[NDifferentDelays[i]];
			IndexSynapseDelay[i]=(int *) new int [NDifferentDelays[i]]();
			
			int index=0;
			delay1=neuron->GetOutputConnectionAt(i,0)->GetDelay();

			NSynapsesWithEqualDelay[i][index]++;
			SynapseDelay[i][index]=delay1;
			OutputConnectionsWithEquealDealy[i][index]=neuron->GetOutputConnectionAt(i,0);

			for(int j=1; j<neuron->GetOutputNumber(i); j++){
				delay2=neuron->GetOutputConnectionAt(i,j)->GetDelay();
				if(delay1!=delay2){
					delay1=delay2;
					index++;
					NSynapsesWithEqualDelay[i][index]++;
					SynapseDelay[i][index]=delay1;
					OutputConnectionsWithEquealDealy[i][index]=neuron->GetOutputConnectionAt(i,j);
				}else{
					NSynapsesWithEqualDelay[i][index]++;
				}
			}
		}
	}
}

NeuronPropagationDelayStructure::~NeuronPropagationDelayStructure(){
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		if(NDifferentDelays[i]!=0){
			delete NSynapsesWithEqualDelay[i];
			delete SynapseDelay[i];
			delete OutputConnectionsWithEquealDealy[i];
			delete IndexSynapseDelay[i];
		}
	}
	delete NDifferentDelays;
	delete NSynapsesWithEqualDelay;
	delete SynapseDelay;
	delete OutputConnectionsWithEquealDealy;
	delete IndexSynapseDelay;
}


void NeuronPropagationDelayStructure::CalculateOutputDelayIndex(NeuronModelPropagationDelayStructure * PropagationStructure){
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		for(int j=0; j<NDifferentDelays[i]; j++){
			int z=0;
			while(SynapseDelay[i][j]!=PropagationStructure->GetDelayAt(i,z)){
				z++;
			}
			IndexSynapseDelay[i][j]=z;
		}
	}
}
