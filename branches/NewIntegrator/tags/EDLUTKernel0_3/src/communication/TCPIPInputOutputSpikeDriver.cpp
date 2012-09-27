/***************************************************************************
 *                           TCPIPInputOutputSpikeDriver.cpp               *
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

#include "../../include/communication/TCPIPInputOutputSpikeDriver.h"

#include "../../include/communication/CdSocket.h"

#include "../../include/simulation/EventQueue.h"

#include "../../include/spike/InputSpike.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"

TCPIPInputOutputSpikeDriver::TCPIPInputOutputSpikeDriver(CdSocket * NewSocket):Socket(NewSocket){
	this->Finished = false;	
}
		
TCPIPInputOutputSpikeDriver::~TCPIPInputOutputSpikeDriver(){
	
}

void TCPIPInputOutputSpikeDriver::LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException){
	unsigned short csize;
	
	this->Socket->receiveBuffer(&csize, sizeof(unsigned short));
	
	if (csize>0){
		OutputSpike * InputSpikes = new OutputSpike [csize];
	
		this->Socket->receiveBuffer(InputSpikes,sizeof(OutputSpike)*(int) csize);
		
		for (int c=0; c<csize; ++c){
			InputSpike * NewSpike = new InputSpike(InputSpikes[c].Time, Net->GetNeuronAt(InputSpikes[c].Neuron));
						
			Queue->InsertEvent(NewSpike);				
		}
	}
}

	
void TCPIPInputOutputSpikeDriver::WriteSpike(const Spike * NewSpike) throw (EDLUTException){
	OutputSpike spike(NewSpike->GetSource()->GetIndex(),NewSpike->GetTime());
		
	this->OutputBuffer.push_back(spike);	
}
		
void TCPIPInputOutputSpikeDriver::WritePotential(float Time, Neuron * Source, float Value) throw (EDLUTException){
	return;	
}
		
bool TCPIPInputOutputSpikeDriver::IsBuffered() const{
	return true;
}

bool TCPIPInputOutputSpikeDriver::IsWritePotentialCapable() const{
	return false;
}
		 
void TCPIPInputOutputSpikeDriver::FlushBuffers() throw (EDLUTException){
	
	unsigned short size = (unsigned short)this->OutputBuffer.size();
	this->Socket->sendBuffer(&size,sizeof(unsigned short));
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
