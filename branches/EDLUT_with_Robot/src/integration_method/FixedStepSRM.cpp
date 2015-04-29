/***************************************************************************
 *                           FixedStepSRM.cpp                              *
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

#include "../../include/integration_method/FixedStepSRM.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"


FixedStepSRM::FixedStepSRM():IntegrationMethod(NULL, "FixedStepSRM",0,0,0,false,false){

}

FixedStepSRM::~FixedStepSRM(){

}


void FixedStepSRM::loadParameter(FILE *fh, long * Currentline) throw (EDLUTFileException){

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%lf",&ElapsedTime)==1){
		if(ElapsedTime<=0.0){
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

ostream & FixedStepSRM::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	