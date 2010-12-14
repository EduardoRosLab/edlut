/***************************************************************************
 *                           TCPIPOutputSpikeDriver.cpp                     *
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

#include "../../include/communication/TCPIPOutputSpikeDriver.h"
#include "../../include/communication/CdSocket.h"

#include "../../include/spike/Spike.h"
#include "../../include/spike/Neuron.h"


TCPIPOutputSpikeDriver::TCPIPOutputSpikeDriver(CdSocket * NewSocket):Socket(NewSocket){
	
}
		
TCPIPOutputSpikeDriver::~TCPIPOutputSpikeDriver(){
	
}
	
void TCPIPOutputSpikeDriver::WriteSpike(const Spike * NewSpike) throw (EDLUTException){
	OutputSpike spike(NewSpike->GetSource()->GetIndex(),NewSpike->GetTime());
		
	this->OutputBuffer.push_back(spike);	
}
		
void TCPIPOutputSpikeDriver::WritePotential(float Time, Neuron * Source, float Value) throw (EDLUTException){
	return;	
}
		
bool TCPIPOutputSpikeDriver::IsBuffered() const{
	return true;
}

bool TCPIPOutputSpikeDriver::IsWritePotentialCapable() const{
	return false;
}
		 
void TCPIPOutputSpikeDriver::FlushBuffers() throw (EDLUTException){
	
	unsigned short size = (unsigned short)this->OutputBuffer.size();
	this->Socket->sendBuffer(&size,sizeof(unsigned short));
	cout << "Tamaño enviado: " << size << endl;
	if (size>0){
		OutputSpike * Array = new OutputSpike [size];
		
		for (unsigned short int i=0; i<size; ++i){
			Array[i] = this->OutputBuffer[i];	
		}
		
		this->Socket->sendBuffer((char *) Array, sizeof(OutputSpike)*size);
					
		delete [] Array;
		
		this->OutputBuffer.clear();
	}
}		
