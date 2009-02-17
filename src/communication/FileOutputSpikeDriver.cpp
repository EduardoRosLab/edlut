#include "./include/FileOutputSpikeDriver.h"

#include "../spike/include/Spike.h"
#include "../spike/include/Neuron.h"

FileOutputSpikeDriver::FileOutputSpikeDriver(const char * NewFileName) throw (EDLUTException): FileName(NewFileName){
	this->Handler = fopen(NewFileName,"w");
	if (!this->Handler){
		throw EDLUTException(2,2,2,0);
	}
}
		
FileOutputSpikeDriver::~FileOutputSpikeDriver(){
	fclose(this->Handler);
}

void FileOutputSpikeDriver::WriteSpike(const Spike * NewSpike) throw (EDLUTException){
	if(!fprintf(this->Handler,"%f %li 1.0\n",NewSpike->GetTime(),NewSpike->GetSource()->GetIndex())>0)
    	throw EDLUTException(3,3,2,0);
}
		
void FileOutputSpikeDriver::WritePotential(float Time, Neuron * Source, float Value) throw (EDLUTException){
	if(!fprintf(this->Handler,"%f %li %f\n",Time,Source->GetIndex(),Value) > 0)
		throw EDLUTException(3,3,2,0);
}

bool FileOutputSpikeDriver::IsBuffered() const{
	return false;	
}

bool FileOutputSpikeDriver::IsWritePotentialCapable() const{
	return true;	
}

void FileOutputSpikeDriver::FlushBuffers() throw (EDLUTException){
	return;
}
	
