/***************************************************************************
 *                           FileOutputSpikeDriver.cpp                     *
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
 
#include "../../include/communication/FileOutputSpikeDriver.h"

#include "../../include/neuron_model/NeuronState.h"

#include "../../include/spike/Spike.h"
#include "../../include/spike/Neuron.h"

FileOutputSpikeDriver::FileOutputSpikeDriver(const char * NewFileName, bool WritePotential) throw (EDLUTException): FileName(NewFileName){
	this->PotentialWriteable = WritePotential;
	this->Handler = fopen(NewFileName,"wt");
	if (!this->Handler){
		throw EDLUTException(2,2,2,0);
	}
}
		
FileOutputSpikeDriver::~FileOutputSpikeDriver(){
	cout << "Estamos en el destructor de file" << endl;
	fclose(this->Handler);
}

void FileOutputSpikeDriver::WriteSpike(const Spike * NewSpike) throw (EDLUTException){
	if(!fprintf(this->Handler,"%f\t%li\n",NewSpike->GetTime(),NewSpike->GetSource()->GetIndex())>0)
    	throw EDLUTException(3,3,2,0);
}
		
void FileOutputSpikeDriver::WriteState(float Time, Neuron * Source) throw (EDLUTException){
	if(!fprintf(this->Handler,"%f\t%li",Time,Source->GetIndex()) > 0)
		throw EDLUTException(3,3,2,0);

	for (unsigned int i=0; i<Source->GetNeuronState()->GetNumberOfPrintableValues(); ++i){
		if(!fprintf(this->Handler,"\t%f",Source->GetNeuronState()->GetPrintableValuesAt(i)) > 0)
				throw EDLUTException(3,3,2,0);
	}

	if(!fprintf(this->Handler,"\n") > 0)
		throw EDLUTException(3,3,2,0);
}

bool FileOutputSpikeDriver::IsBuffered() const{
	return false;	
}

bool FileOutputSpikeDriver::IsWritePotentialCapable() const{
	return this->PotentialWriteable;	
}

void FileOutputSpikeDriver::FlushBuffers() throw (EDLUTException){
	return;
}

ostream & FileOutputSpikeDriver::PrintInfo(ostream & out){

	out << "- File Output Spike Driver: " << this->FileName << endl;

	if (this->PotentialWriteable) out << "\tWriteable Potential" << endl;
	else out << "\tNon-writeable Potential" << endl;

	return out;
}
	
