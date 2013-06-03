/***************************************************************************
 *                           AdditiveKernelChange.cpp                      *
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

#include "../../include/learning_rules/AdditiveKernelChange.h"

#include "../../include/learning_rules/ExpState.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include <cmath>

int AdditiveKernelChange::GetNumberOfVar() const{
	return 2;
}

void AdditiveKernelChange::LoadLearningRule(FILE * fh, long & Currentline) throw (EDLUTFileException){
	skip_comments(fh,Currentline);

	if(fscanf(fh,"%i",&this->trigger)==1 && fscanf(fh,"%f",&this->maxpos)==1 && fscanf(fh,"%f",&this->a1pre)==1 && fscanf(fh,"%f",&this->a2prepre)==1){
		if(this->a1pre < -1.0 || this->a1pre > 1.0){
			throw EDLUTFileException(4,27,22,1,Currentline);
		}

		this->numexps = 3;
	}else{
		throw EDLUTFileException(4,28,23,1,Currentline);
	}
}

void AdditiveKernelChange::ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime){

	// Second case: the weight change is linked to this connection
	float NewWeight = Connection->GetWeight()+this->a1pre;

	if(NewWeight>Connection->GetMaxWeight())
		NewWeight=Connection->GetMaxWeight();
	else if(NewWeight<0.0)
		NewWeight=0.0;

	//Connection->SetLastSpikeTime(SpikeTime);
	Connection->SetWeight(NewWeight);

	// Get connection state
	ConnectionState * State = Connection->GetConnectionState();

	// Update the presynaptic activity
	State->AddElapsedTime(SpikeTime-State->GetLastUpdateTime());

	// Add the presynaptic spike influence
	State->ApplyPresynapticSpike();

	// Check if this is the teaching signal
	if(this->trigger == 1){
		for(int i=0; i<Connection->GetTarget()->GetInputNumberWithLearning(); ++i){
			Interconnection * interi=Connection->GetTarget()->GetInputConnectionWithLearningAt(i);
		    AdditiveKernelChange * wchani=(AdditiveKernelChange *)interi->GetWeightChange();

		    // Apply sinaptic plasticity driven by teaching signal
		    // Get connection state
			ConnectionState * ConnectionStatePre = interi->GetConnectionState();

			// Update the presynaptic activity
			ConnectionStatePre->AddElapsedTime(SpikeTime-ConnectionStatePre->GetLastUpdateTime());

			// Update synaptic weight
			float NewWeightPre = interi->GetWeight()+wchani->a2prepre*ConnectionStatePre->GetPresynapticActivity();

			if(NewWeightPre>interi->GetMaxWeight())
				NewWeightPre=interi->GetMaxWeight();
			else if(NewWeightPre<0.0)
				NewWeightPre=0.0;

			interi->SetWeight(NewWeightPre);
		}
	}
}

void AdditiveKernelChange::ApplyPostSynapticSpike(Interconnection * Connection,double SpikeTime){
	return;
}



ostream & AdditiveKernelChange::PrintInfo(ostream & out){

	out << "- Additive Kernel Learning Rule: " << this->trigger << "\t" << this->maxpos << "\t" << this->a1pre << "\t" << this->a2prepre << endl;


	return out;
}
