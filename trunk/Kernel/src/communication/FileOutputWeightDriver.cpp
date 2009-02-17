#include "./include/FileOutputWeightDriver.h"

#include "../spike/include/Network.h"

FileOutputWeightDriver::FileOutputWeightDriver(const char * NewFileName) throw (EDLUTException): FileName(NewFileName){
}
		
FileOutputWeightDriver::~FileOutputWeightDriver(){
}

void FileOutputWeightDriver::WriteWeights(Network * Net, float SimulationTime) throw (EDLUTException){
	string Name = FileName;
	
	char* str = new char[30];
    sprintf(str, "%.4g", SimulationTime );    

	Name = Name.insert(Name.find_last_of('.'),string(str));		
	
	Net->SaveWeights(Name.c_str());
	
	return;
}	

void FileOutputWeightDriver::WriteWeights(Network * Net) throw (EDLUTException){
	Net->SaveWeights(FileName);
}	

