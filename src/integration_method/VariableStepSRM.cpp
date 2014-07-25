/***************************************************************************
 *                           VariableStepSRM.cpp                           *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
 * email                : fnaveros@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/integration_method/VariableStepSRM.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"


VariableStepSRM::VariableStepSRM():IntegrationMethod(NULL, "VariableStepSRM",0,0,0,false,false){
}

VariableStepSRM::~VariableStepSRM(){
}


void VariableStepSRM::loadParameter(FILE *fh, long * Currentline) throw (EDLUTFileException){
	this->PredictedElapsedTime=new double [1];

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%lf",PredictedElapsedTime)==1){
		if(PredictedElapsedTime[0]<=0.0){
////NEW CODE------------------------------------------------------------------------------
			throw EDLUTFileException(4,7,6,1,*Currentline);
////--------------------------------------------------------------------------------------
		}
	}else{
//NEW CODE------------------------------------------------------------------------------
		throw EDLUTFileException(4,7,6,1,*Currentline);
//--------------------------------------------------------------------------------------
	}

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%f",&h_max)==1){
		if(h_max<=0.0){
////NEW CODE------------------------------------------------------------------------------
			throw EDLUTFileException(4,7,6,1,*Currentline);
////--------------------------------------------------------------------------------------
		}
	}else{
//NEW CODE------------------------------------------------------------------------------
		throw EDLUTFileException(4,7,6,1,*Currentline);
//--------------------------------------------------------------------------------------
	}

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%f",&h_min)==1){
		if(h_min<=0.0){
////NEW CODE------------------------------------------------------------------------------
			throw EDLUTFileException(4,7,6,1,*Currentline);
////--------------------------------------------------------------------------------------
		}
	}else{
//NEW CODE------------------------------------------------------------------------------
		throw EDLUTFileException(4,7,6,1,*Currentline);
//--------------------------------------------------------------------------------------
	}

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%f",&p_max)==1){
		if(p_max<=0.0){
////NEW CODE------------------------------------------------------------------------------
			throw EDLUTFileException(4,7,6,1,*Currentline);
////--------------------------------------------------------------------------------------
		}
	}else{
//NEW CODE------------------------------------------------------------------------------
		throw EDLUTFileException(4,7,6,1,*Currentline);
//--------------------------------------------------------------------------------------
	}

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%f",&p_min)==1){
		if(p_min<=0.0){
////NEW CODE------------------------------------------------------------------------------
			throw EDLUTFileException(4,7,6,1,*Currentline);
////--------------------------------------------------------------------------------------
		}
	}else{
//NEW CODE------------------------------------------------------------------------------
		throw EDLUTFileException(4,7,6,1,*Currentline);
//--------------------------------------------------------------------------------------
	}

}

enum IntegrationMethodType VariableStepSRM::GetMethodType(){
	return VARIABLE_STEP;
}


void VariableStepSRM::InitializeStates(int N_neurons, float * initialization){
	float elapsedTime=PredictedElapsedTime[0];
	delete [] PredictedElapsedTime;
	PredictedElapsedTime=new double[N_neurons];

	for(int i=0; i<N_neurons; i++){
		PredictedElapsedTime[i]=elapsedTime;
	}
}


ostream & VariableStepSRM::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	


void VariableStepSRM::NextDifferentialEcuationValue(int index, float * NeuronState, float elapsed_time) {
	float p=NeuronState[4];

	if(p<p_min){
		PredictedElapsedTime[index]*=2;
	}
	if(p>p_max){
		PredictedElapsedTime[index]*=0.5f;
	}

	if(PredictedElapsedTime[index]>h_max){
		PredictedElapsedTime[index]=h_max;
	}
	if(PredictedElapsedTime[index]<h_min){
		PredictedElapsedTime[index]=h_min;
	}

}