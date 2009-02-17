#include "./include/TCPIPOutputSpikeDriver.h"
#include "./include/CdSocket.h"

#include "../spike/include/Spike.h"
#include "../spike/include/Neuron.h"


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
