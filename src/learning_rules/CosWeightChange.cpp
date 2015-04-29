/***************************************************************************
 *                           CosWeightChange.cpp                           *
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


#include "../../include/learning_rules/CosWeightChange.h"

#include "../../include/learning_rules/CosState.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

CosWeightChange::CosWeightChange(int NewLearningRuleIndex):WithoutPostSynaptic(NewLearningRuleIndex){
}

CosWeightChange::~CosWeightChange(){

}


void CosWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses){
	this->State=(ConnectionState *) new CosState(NumberOfSynapses, this->tau, this->exponent);
}


void CosWeightChange::LoadLearningRule(FILE * fh, long & Currentline) throw (EDLUTFileException){
	skip_comments(fh,Currentline);

	if(fscanf(fh,"%f",&this->tau)==1 && fscanf(fh,"%f",&this->exponent)==1 && fscanf(fh,"%f",&this->a1pre)==1 && fscanf(fh,"%f",&this->a2prepre)==1){
		if(this->a1pre < -1.0 || this->a1pre > 1.0){
			throw EDLUTFileException(4,27,22,1,Currentline);
		}
	}else{
		throw EDLUTFileException(4,28,23,1,Currentline);
	}
}


void CosWeightChange::ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime){


	if(Connection->GetTriggerConnection()==false){
		int LearningRuleIndex = Connection->GetLearningRuleIndex_withoutPost();

		// Second case: the weight change is linked to this connection
		Connection->IncrementWeight(this->a1pre);

		// Update the presynaptic activity
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

		// Add the presynaptic spike influence
		State->ApplyPresynapticSpike(LearningRuleIndex);
	
	}else{
		Neuron * TargetNeuron=Connection->GetTarget();

		int aux=(TargetNeuron->GetInputNumberWithoutPostSynapticLearning()+NumberOfOpenMPThreads-1)/NumberOfOpenMPThreads;
		int start, end;
		for(int j=0; j<NumberOfOpenMPThreads; j++){
			start=j*aux;
			end=start+aux;
			if(end>TargetNeuron->GetInputNumberWithoutPostSynapticLearning()){
				end=TargetNeuron->GetInputNumberWithoutPostSynapticLearning();
			}

			#ifdef _OPENMP 
				#if	_OPENMP >= OPENMPVERSION30
					#pragma omp task if(j<(NumberOfOpenMPThreads-1)) firstprivate(start,end) shared(TargetNeuron, Connection)
				#endif
			#endif
			{
				for(int i=start; i<end; ++i){
					Interconnection * interi=TargetNeuron->GetInputConnectionWithoutPostSynapticLearningAt(i);

					if(interi->GetTriggerConnection()==false){
						// Apply sinaptic plasticity driven by teaching signal
						int LearningRuleIndex = interi->GetLearningRuleIndex_withoutPost();

						// Update the presynaptic activity
						State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);
						// Update synaptic weight
						interi->IncrementWeight(this->a2prepre*State->GetPresynapticActivity(LearningRuleIndex));
					}
				}
			}
		}
		#ifdef _OPENMP 
			#if	_OPENMP >= OPENMPVERSION30
				#pragma omp taskwait
			#endif
		#endif
	}
}

ostream & CosWeightChange::PrintInfo(ostream & out){

	out << "- Cos Kernel Learning Rule: \t" << this->tau << "\t" << this->a1pre << "\t" << this->a2prepre << endl;


	return out;
}







