#include "./include/TCPIPInputSpikeDriver.h"

#include "./include/CdSocket.h"

#include "../simulation/include/EventQueue.h"

#include "../spike/include/InputSpike.h"
#include "../spike/include/Network.h"



TCPIPInputSpikeDriver::TCPIPInputSpikeDriver(CdSocket * NewSocket): Socket(NewSocket){
	this->Finished = false;
}
		
TCPIPInputSpikeDriver::~TCPIPInputSpikeDriver(){
}
	
void TCPIPInputSpikeDriver::LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException){
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
