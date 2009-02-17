#include "./include/FileInputSpikeDriver.h"

#include "../simulation/include/utils.h"
#include "../simulation/include/EventQueue.h"

#include "../spike/include/EDLUTFileException.h"
#include "../spike/include/Network.h"
#include "../spike/include/InputSpike.h"

FileInputSpikeDriver::FileInputSpikeDriver(const char * NewFileName) throw (EDLUTException): FileName(NewFileName), Currentline(1L){
	this->Finished = false;
	this->Handler = fopen(NewFileName,"rt");
	if (!this->Handler){
		throw EDLUTException(6,20,13,0);
	}
}
		
FileInputSpikeDriver::~FileInputSpikeDriver(){
	fclose(this->Handler);
}
	
void FileInputSpikeDriver::LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException){
	if (this->Handler){
		int ninputs,i,ret;
		skip_comments(this->Handler,Currentline);
		if(fscanf(this->Handler,"%i",&ninputs)==1){
			int nspikes,nneuron,nreps,ineuron,itime;
			float time,interv;
			ret=0;
			
			for(i=0;i<ninputs;){
				skip_comments(this->Handler,Currentline);
				if(fscanf(this->Handler,"%f",&time)==1 && fscanf(this->Handler,"%i",&nspikes)==1 && fscanf(this->Handler,"%f",&interv)==1 && fscanf(this->Handler,"%i",&nneuron)==1 && fscanf(this->Handler,"%i",&nreps)==1){
					if(nneuron+nreps<=Net->GetNeuronNumber() && nneuron >= 0){
						i+=nspikes*nreps;
						
						if(i<=ninputs){
							for(itime=0;itime<nspikes;itime++){
								for(ineuron=0;ineuron<nreps;ineuron++){
									InputSpike * ispike = new InputSpike(time+itime*interv, Net->GetNeuronAt(nneuron+ineuron));
									
									Queue->InsertEvent(ispike);
								}
							}
						}else{
							throw EDLUTFileException(6,16,15,1,Currentline);
						}
					}else{
						throw EDLUTFileException(6,17,16,1,Currentline);
						break;
					}
				}else{
					throw EDLUTFileException(6,18,17,1,Currentline);
					break;
				}
			}			
		}else{
			throw EDLUTFileException(6,19,18,1,Currentline);
		}		
		
		this->Finished = true;
	}
	
}
