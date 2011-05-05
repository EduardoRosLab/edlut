/***************************************************************************
 *                           MultiplicativeKernelChange.cpp                *
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

#include "../../include/learning_rules/MultiplicativeKernelChange.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include <cmath>

int MultiplicativeKernelChange::GetNumberOfVar() const{
	return 2;
}

void MultiplicativeKernelChange::LoadLearningRule(FILE * fh, long & Currentline) throw (EDLUTFileException){
	skip_comments(fh,Currentline);

	if(fscanf(fh,"%i",&this->trigger)==1 && fscanf(fh,"%f",&this->maxpos)==1 && fscanf(fh,"%f",&this->a1pre)==1 && fscanf(fh,"%f",&this->a2prepre)==1){
		if(this->a1pre < -1.0 || this->a1pre > 1.0){
			throw EDLUTFileException(4,27,22,1,Currentline);
		}

		this->numexps = 3;
	}else{
		throw EDLUTFileException(4,28,23,1,Currentline);
	}

	static float explpar[]={30.1873,60.3172,5.9962};
	static float expcpar[]={-5.2410,3.1015,2.2705};


	for(int indexp=0;indexp<this->numexps;indexp++){
		this->SetLparAt(indexp,(this->maxpos == 0)?0:(0.1/this->maxpos)*explpar[indexp]);
		this->SetCparAt(indexp,expcpar[indexp]);
	}
}

void MultiplicativeKernelChange::ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime){

	int indexp;
	float activ;
	MultiplicativeKernelChange *wchani;

	for(indexp=0;indexp<this->numexps;indexp++){
		Connection->SetActivityAt(indexp,1+Connection->GetActivityAt(indexp)*exp((Connection->GetLastSpikeTime()-SpikeTime)*this->GetLparAt(indexp)));
	}

	//Connection->SetLastSpikeTime(SpikeTime);
	Connection->SetWeight(Connection->GetWeight() + this->a1pre*((this->a1pre > 0.0)?(Connection->GetMaxWeight()-Connection->GetWeight()):Connection->GetWeight()));

	if(this->trigger == 1){
		Interconnection *interi;
		for(int i=0; i<Connection->GetTarget()->GetInputNumber(); ++i){
			interi=Connection->GetTarget()->GetInputConnectionAt(i);
			wchani=(MultiplicativeKernelChange *) interi->GetWeightChange();
			if (wchani!=0){
				activ=0;
				for(indexp=0;indexp<wchani->numexps;indexp++){
					activ+=wchani->GetCparAt(indexp)*interi->GetActivityAt(indexp)*exp((interi->GetLastSpikeTime()-SpikeTime)*wchani->GetLparAt(indexp));
				}
				interi->SetWeight(interi->GetWeight() + wchani->a2prepre*activ*((wchani->a2prepre > 0.0)?(interi->GetMaxWeight()-interi->GetWeight()):interi->GetWeight()));
			}
        }
	}
}

void MultiplicativeKernelChange::ApplyPostSynapticSpike(Interconnection * Connection,double SpikeTime){
	return;
}

float MultiplicativeKernelChange::GetLparAt(int index) const{
	return this->lpar[index];
}

void MultiplicativeKernelChange::SetLparAt(int index, float NewLpar){
	this->lpar[index] = NewLpar;
}

float MultiplicativeKernelChange::GetCparAt(int index) const{
	return this->cpar[index];
}

void MultiplicativeKernelChange::SetCparAt(int index, float NewCpar){
	this->cpar[index] = NewCpar;
}


ostream & MultiplicativeKernelChange::PrintInfo(ostream & out){

	out << "- Multiplicative Kernel Learning Rule: " << this->trigger << "\t" << this->maxpos << "\t" << this->a1pre << "\t" << this->a2prepre << endl;

	return out;
}
