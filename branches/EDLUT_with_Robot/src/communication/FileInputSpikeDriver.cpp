/***************************************************************************
 *                           FileInputSpikeDriver.cpp                      *
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
 
#include "../../include/communication/FileInputSpikeDriver.h"

#include "../../include/simulation/Utils.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/InputSpike.h"
#include "../../include/spike/Neuron.h"

#include "../../include/openmp/openmp.h" /*revisar*/
#include "../../include/spike/InputSpikeGroupe.h"


FileInputSpikeDriver::FileInputSpikeDriver(const char * NewFileName) throw (EDLUTException): FileName(NewFileName), Currentline(1L){
	this->Finished = false;
	this->Handler = fopen(NewFileName,"rt");
	if (!this->Handler){
		throw EDLUTException(6,20,13,0);
	}
}
		
FileInputSpikeDriver::~FileInputSpikeDriver(){
	if (this->Handler){
		fclose(this->Handler);
		this->Handler=NULL;
	}
}
	
void FileInputSpikeDriver::LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException){
	if (this->Handler){
		int ninputs,i;
		skip_comments(this->Handler,Currentline);
		if(fscanf(this->Handler,"%i",&ninputs)==1){
			int nspikes,nneuron,nreps,ineuron,itime;
			float time,interv;
			
			for(i=0;i<ninputs;){
				skip_comments(this->Handler,Currentline);
				if(fscanf(this->Handler,"%f",&time)==1 && fscanf(this->Handler,"%i",&nspikes)==1 && fscanf(this->Handler,"%f",&interv)==1 && fscanf(this->Handler,"%i",&nneuron)==1 && fscanf(this->Handler,"%i",&nreps)==1){
					if(nneuron+nreps<=Net->GetNeuronNumber() && nneuron >= 0){
						i+=nspikes*nreps;
						
						if(i<=ninputs){
							for(itime=0;itime<nspikes;itime++){
								for(ineuron=0;ineuron<nreps;ineuron++){
									InputSpike * ispike = new InputSpike(time+itime*interv, Net->GetNeuronAt(nneuron+ineuron));
									Queue->InsertEvent(Net->GetNeuronAt(nneuron+ineuron)->get_OpenMP_queue_index(), ispike);
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

//void FileInputSpikeDriver::LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException){
//	if (this->Handler){
//		int ninputs,i;
//		skip_comments(this->Handler,Currentline);
//		if(fscanf(this->Handler,"%i",&ninputs)==1){
//			int nspikes,nneuron,nreps,ineuron,itime;
//			float time,interv;
/////*revisar*/
//int MaxSize=1024*16;
//float * Times=new float[NumberOfOpenMPQueues]();
//int * NElements=new int[NumberOfOpenMPQueues]();
//Neuron *** NeuronIndex=(Neuron***)new Neuron**[NumberOfOpenMPQueues];
//for(int j=0; j<NumberOfOpenMPQueues; j++){
//	NeuronIndex[j]=(Neuron**)new Neuron*[MaxSize];
//}
//Neuron * source;
//int OpenMPQueueIndex;
/////*revisar*/			
//			for(i=0;i<ninputs;){
//				skip_comments(this->Handler,Currentline);
//				if(fscanf(this->Handler,"%f",&time)==1 && fscanf(this->Handler,"%i",&nspikes)==1 && fscanf(this->Handler,"%f",&interv)==1 && fscanf(this->Handler,"%i",&nneuron)==1 && fscanf(this->Handler,"%i",&nreps)==1){
//					if(nneuron+nreps<=Net->GetNeuronNumber() && nneuron >= 0){
//						i+=nspikes*nreps;
//						
//						if(i<=ninputs){
//							for(itime=0;itime<nspikes;itime++){
//								for(ineuron=0;ineuron<nreps;ineuron++){
//									source=Net->GetNeuronAt(nneuron+ineuron);
//									OpenMPQueueIndex=source->get_OpenMP_queue_index();
//									if(Times[OpenMPQueueIndex]!=time+itime*interv || NElements[OpenMPQueueIndex]==MaxSize){
//										if(NElements[OpenMPQueueIndex]>0){
//											Neuron ** NewSources=(Neuron**)new Neuron*[NElements[OpenMPQueueIndex]];
//											memcpy(NewSources, NeuronIndex[OpenMPQueueIndex],sizeof(Neuron*)*NElements[OpenMPQueueIndex]);
//
//											InputSpikeGroupe * ispike = new InputSpikeGroupe(Times[OpenMPQueueIndex], NewSources, NElements[OpenMPQueueIndex]);
//											Queue->InsertEvent(OpenMPQueueIndex,ispike);
//										}
//										Times[OpenMPQueueIndex]=time+itime*interv;
//										NElements[OpenMPQueueIndex]=0;
//									}
//
//									NeuronIndex[OpenMPQueueIndex][NElements[OpenMPQueueIndex]]=source;
//									NElements[OpenMPQueueIndex]++;
//								}
//							}
//						}else{
//							throw EDLUTFileException(6,16,15,1,Currentline);
//						}
//					}else{
//						throw EDLUTFileException(6,17,16,1,Currentline);
//						break;
//					}
//				}else{
//					throw EDLUTFileException(6,18,17,1,Currentline);
//					break;
//				}
//			}
//
//			for(int j=0; j<NumberOfOpenMPQueues; j++){
//				if(NElements[j]>0){
//					Neuron ** NewSources=(Neuron**)new Neuron*[NElements[j]];
//					memcpy(NewSources, NeuronIndex[j],sizeof(Neuron*)*NElements[j]);
//
//					InputSpikeGroupe * ispike = new InputSpikeGroupe(Times[j], NewSources, NElements[j]);
//					Queue->InsertEvent(j,ispike);
//				}
//				delete NeuronIndex[j];
//			}
//			delete NeuronIndex;
//			delete Times;
//			delete NElements;
//
//		}else{
//			throw EDLUTFileException(6,19,18,1,Currentline);
//		}		
//		
//		this->Finished = true;
//	}
//}

ostream & FileInputSpikeDriver::PrintInfo(ostream & out){

	out << "- File Input Spike Driver: " << this->FileName << endl;

	return out;
}
