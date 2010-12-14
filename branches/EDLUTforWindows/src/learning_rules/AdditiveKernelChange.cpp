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

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include <cmath>

int AdditiveKernelChange::GetNumberOfVar() const{
	return 2;
}

void AdditiveKernelChange::update_activity(double time,Interconnection * Connection,bool spike){
	// CHANGED
	// VERSION USING ANALYTICALLY SOLVED EQUATIONS
	float delta_t = (time-Connection->GetLastSpikeTime());
	float tau = this->maxpos;
	if (tau==0){
		tau = 1e-6;
	}
	float quot = delta_t/tau;
	float ex = exp(-quot);

	float OldE1 = Connection->GetActivityAt(1);
	float OldE = Connection->GetActivityAt(0);
	float NewE = (OldE+quot*OldE1)*ex;
	float NewE1 = OldE1*ex;

	if(spike){  // if spike, we need to increase the e1 variable
		NewE1 += 1;
	}

	Connection->SetActivityAt(1,NewE1);
	Connection->SetActivityAt(0,NewE);
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

	// CHANGED
	this->update_activity(SpikeTime,Connection,true);

	if(NewWeight>Connection->GetMaxWeight())
		NewWeight=Connection->GetMaxWeight();
	else if(NewWeight<0.0)
		NewWeight=0.0;

	Connection->SetLastSpikeTime(SpikeTime);
	Connection->SetWeight(NewWeight);

	if(this->trigger == 1){
		for(int i=0; i<Connection->GetTarget()->GetInputNumber(); ++i){
			Interconnection * interi=Connection->GetTarget()->GetInputConnectionAt(i);
		    AdditiveKernelChange * wchani=(AdditiveKernelChange *)interi->GetWeightChange();
		    if (wchani!=0){
		    	//CHANGED
		     	wchani->update_activity(SpikeTime,interi, false);
		     	float NewWeight = interi->GetWeight()+wchani->a2prepre*interi->GetActivityAt(0);
		     	//
		     	if(NewWeight>interi->GetMaxWeight())

		     		NewWeight=interi->GetMaxWeight();
		     	else if(NewWeight<0.0)
		       		NewWeight=0.0;

		       	interi->SetWeight(NewWeight);
		    }
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
