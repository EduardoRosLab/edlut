/***************************************************************************
 *                           Network.cpp                                   *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido, Richard Carrillo and  *
 *						: Francisco Naveros                                *
 * email                : jgarrido@atc.ugr.es, fnaveros@atc.ugr.es         *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/spike/Network.h"
#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"

#include "../../include/learning_rules/ExpWeightChange.h"
#include "../../include/learning_rules/SinWeightChange.h"
#include "../../include/learning_rules/CosWeightChange.h"
#include "../../include/learning_rules/SimetricCosWeightChange.h"
#include "../../include/learning_rules/STDPWeightChange.h"
#include "../../include/learning_rules/STDPLSWeightChange.h"

#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/SRMTimeDrivenModel.h"
#include "../../include/neuron_model/LIFTimeDrivenModel_1_4.h"
#include "../../include/neuron_model/LIFTimeDrivenModel_1_2.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"
#include "../../include/neuron_model/EventDrivenNeuronModel.h"
#include "../../include/neuron_model/TableBasedModel.h"
#include "../../include/neuron_model/SRMTableBasedModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/EgidioGranuleCell_TimeDriven.h"
#include "../../include/neuron_model/Vanderpol.h"

#include "../../include/simulation/EventQueue.h"
#include "../../include/simulation/Utils.h"
#include "../../include/simulation/Configuration.h"

int qsort_inters(const void *e1, const void *e2){
	int ord;
	float ordf;
	
	ord=((Interconnection *)e1)->GetSource()->GetIndex() - ((Interconnection *)e2)->GetSource()->GetIndex();
	if(!ord){
		ordf=((Interconnection *)e1)->GetDelay() - ((Interconnection *)e2)->GetDelay();
		if(ordf<0.0)
			ord=-1;
		else
			if(ordf>0.0)
				ord=1;
	}
   
	return(ord);
}

void Network::FindOutConnections(){
	// Change the ordenation
   	qsort(inters,ninters,sizeof(Interconnection),qsort_inters);
	if(ninters>0){
		// Calculate the number of input connections with learning for each cell
		unsigned long * NumberOfOutputs = (unsigned long *) new unsigned long [this->nneurons];
		unsigned long * OutputsLeft = (unsigned long *) new unsigned long [this->nneurons];

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			NumberOfOutputs[neu] = 0;
		}

		for (unsigned long con= 0; con<this->ninters; ++con){
			NumberOfOutputs[this->inters[con].GetSource()->GetIndex()]++;
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			OutputsLeft[neu] = NumberOfOutputs[neu];
		}

		Interconnection *** OutputConnections = (Interconnection ***) new Interconnection ** [this->nneurons];

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			if (NumberOfOutputs[neu]>0){
				OutputConnections[neu] = (Interconnection **) new Interconnection * [NumberOfOutputs[neu]];
			} else {
				OutputConnections[neu] = 0;
			}
		}

		for (unsigned long con= this->ninters-1; con<this->ninters; --con){
			unsigned long SourceCell = this->inters[con].GetSource()->GetIndex();
			OutputConnections[SourceCell][--OutputsLeft[SourceCell]] = this->inters+con;
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			this->neurons[neu].SetOutputConnections(OutputConnections[neu],NumberOfOutputs[neu]);

			for (unsigned long aux = 0; aux < NumberOfOutputs[neu]; aux++){
				if(OutputConnections[neu][aux]->GetWeightChange_withoutPost()!=0){
					OutputConnections[neu][aux]->SetLearningRuleIndex_withoutPost(OutputConnections[neu][aux]->GetWeightChange_withoutPost()->counter);
					OutputConnections[neu][aux]->GetWeightChange_withoutPost()->counter++;
				}
			}
		}

		delete [] OutputConnections;
		delete [] NumberOfOutputs;
		delete [] OutputsLeft;
	}
}

void Network::SetWeightOrdination(){
	if (ninters>0){
		for (int ninter=0; ninter<ninters;ninter++){
			int index = inters[ninter].GetIndex();
			this->wordination[index] = &(this->inters[ninter]);
		}
	}	
}

void Network::FindInConnections(){
	if(this->ninters>0){

		// Calculate the number of input connections with learning for each cell
		unsigned long * NumberOfInputsWithPostSynapticLearning = (unsigned long *) new unsigned long [this->nneurons]();
		unsigned long * InputsLeftWithPostSynapticLearning = (unsigned long *) new unsigned long [this->nneurons];

		unsigned long * NumberOfInputsWithoutPostSynapticLearning = (unsigned long *) new unsigned long [this->nneurons]();
		unsigned long * InputsLeftWithoutPostSynapticLearning = (unsigned long *) new unsigned long [this->nneurons];

		for (unsigned long con= 0; con<this->ninters; ++con){
			if(this->inters[con].GetWeightChange_withPost()!=0){
				NumberOfInputsWithPostSynapticLearning[this->inters[con].GetTarget()->GetIndex()]++;
			}
			if(this->inters[con].GetWeightChange_withoutPost()!=0){
				NumberOfInputsWithoutPostSynapticLearning[this->inters[con].GetTarget()->GetIndex()]++;
			}
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			InputsLeftWithPostSynapticLearning[neu] = NumberOfInputsWithPostSynapticLearning[neu];
			InputsLeftWithoutPostSynapticLearning[neu] = NumberOfInputsWithoutPostSynapticLearning[neu];
		}

		Interconnection *** InputConnectionsWithPostSynapticLearning = (Interconnection ***) new Interconnection ** [this->nneurons];
		Interconnection *** InputConnectionsWithoutPostSynapticLearning = (Interconnection ***) new Interconnection ** [this->nneurons];

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			InputConnectionsWithPostSynapticLearning[neu] = 0;
			InputConnectionsWithoutPostSynapticLearning[neu] = 0;

			if (NumberOfInputsWithPostSynapticLearning[neu]>0){
				InputConnectionsWithPostSynapticLearning[neu] = (Interconnection **) new Interconnection * [NumberOfInputsWithPostSynapticLearning[neu]];
			} 
			if (NumberOfInputsWithoutPostSynapticLearning[neu]>0){
				InputConnectionsWithoutPostSynapticLearning[neu] = (Interconnection **) new Interconnection * [NumberOfInputsWithoutPostSynapticLearning[neu]];
			} 
		}

		for (unsigned long con= this->ninters-1; con<this->ninters; --con){
			if (this->inters[con].GetWeightChange_withPost()!=0){
				unsigned long TargetCell = this->inters[con].GetTarget()->GetIndex();
				InputConnectionsWithPostSynapticLearning[TargetCell][--InputsLeftWithPostSynapticLearning[TargetCell]] = this->inters+con;
			}
			if (this->inters[con].GetWeightChange_withoutPost()!=0){
				unsigned long TargetCell = this->inters[con].GetTarget()->GetIndex();
				InputConnectionsWithoutPostSynapticLearning[TargetCell][--InputsLeftWithoutPostSynapticLearning[TargetCell]] = this->inters+con;
			}
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			this->neurons[neu].SetInputConnectionsWithPostSynapticLearning(InputConnectionsWithPostSynapticLearning[neu],NumberOfInputsWithPostSynapticLearning[neu]);
			this->neurons[neu].SetInputConnectionsWithoutPostSynapticLearning(InputConnectionsWithoutPostSynapticLearning[neu],NumberOfInputsWithoutPostSynapticLearning[neu]);
		
			for (unsigned long aux = 0; aux < NumberOfInputsWithPostSynapticLearning[neu]; aux++){
				InputConnectionsWithPostSynapticLearning[neu][aux]->SetLearningRuleIndex_withPost(InputConnectionsWithPostSynapticLearning[neu][aux]->GetWeightChange_withPost()->counter);
				InputConnectionsWithPostSynapticLearning[neu][aux]->GetWeightChange_withPost()->counter++;
			}

		}

		delete [] InputConnectionsWithPostSynapticLearning;
		delete [] NumberOfInputsWithPostSynapticLearning;
		delete [] InputsLeftWithPostSynapticLearning;

		delete [] InputConnectionsWithoutPostSynapticLearning;
		delete [] NumberOfInputsWithoutPostSynapticLearning;
		delete [] InputsLeftWithoutPostSynapticLearning;
	}
}



NeuronModel * Network::LoadNetTypes(string ident_type, string neutype, int & ni) throw (EDLUTException){
	NeuronModel * type;
   	
   	for(ni=0;ni<nneutypes && neutypes[ni]!=0 && ( neutypes[ni]->GetModelID()==neutype && neutypes[ni]->GetTypeID()!=ident_type || neutypes[ni]->GetModelID()!=neutype);++ni);

   	if (ni<nneutypes && neutypes[ni]==0){
		if (ident_type=="LIFTimeDrivenModel_1_4"){
			neutypes[ni] = (LIFTimeDrivenModel_1_4 *) new LIFTimeDrivenModel_1_4(ident_type, neutype);
		}else if (ident_type=="LIFTimeDrivenModel_1_2"){
			neutypes[ni] = (LIFTimeDrivenModel_1_2 *) new LIFTimeDrivenModel_1_2(ident_type, neutype);
		}else if (ident_type=="SRMTimeDrivenModel"){
			neutypes[ni] = (SRMTimeDrivenModel *) new SRMTimeDrivenModel(ident_type, neutype);
		} else if (ident_type=="TableBasedModel"){
   			neutypes[ni] = (TableBasedModel *) new TableBasedModel(ident_type, neutype);
		} else if (ident_type=="SRMTableBasedModel"){
			neutypes[ni] = (SRMTableBasedModel *) new SRMTableBasedModel(ident_type, neutype);
		} else if (ident_type=="EgidioGranuleCell_TimeDriven"){
			neutypes[ni] = (EgidioGranuleCell_TimeDriven *) new EgidioGranuleCell_TimeDriven(ident_type, neutype);
		}else if (ident_type=="Vanderpol"){
			neutypes[ni] = (Vanderpol *) new Vanderpol(ident_type, neutype);
		}else {
			throw EDLUTException(13,58,30,0);
		}
   		type = neutypes[ni];
   		type->LoadNeuronModel();
   	} else if (ni<nneutypes) {
		type = neutypes[ni];
	} else {
		throw EDLUTException(13,44,20,0);
	}

	return(type);
}

void Network::InitializeStates(int * N_neurons){
	for( int z=0; z< this->nneutypes; z++){
		neutypes[z]->InitializeStates(N_neurons[z]);
	}
}



void Network::InitNetPredictions(EventQueue * Queue){
	int nneu;
	for(nneu=0;nneu<nneurons;nneu++){
		if (neurons[nneu].GetNeuronModel()->GetModelType()==EVENT_DRIVEN_MODEL){
			EventDrivenNeuronModel * Model = (EventDrivenNeuronModel *) neurons[nneu].GetNeuronModel();
			InternalSpike * spike = Model->GenerateInitialActivity(neurons+nneu);
			if (spike!=0){
				Queue->InsertEvent(spike);
			}
		}
	}

}

Network::Network(const char * netfile, const char * wfile, EventQueue * Queue) throw (EDLUTException): inters(0), ninters(0), neutypes(0), nneutypes(0), neurons(0), nneurons(0), timedrivenneurons(0), ntimedrivenneurons(0), wchanges(0), nwchanges(0), wordination(0){
	this->LoadNet(netfile);	
	this->LoadWeights(wfile);
	this->InitNetPredictions(Queue);	
}
   		
Network::~Network(){
	if (inters!=0) {
		delete [] inters;
	}

	if (neutypes!=0) {
		for (int i=0; i<this->nneutypes; ++i){
			if (this->neutypes[i]!=0){
				delete this->neutypes[i];
			}
		}
		delete [] neutypes;
	}
	
	if (neurons!=0) {
		delete [] neurons;
	}

	if (timedrivenneurons!=0) {
		delete [] timedrivenneurons;
	}

	if(ntimedrivenneurons!=0){
		delete [] ntimedrivenneurons;
	}



	if (wchanges!=0) {
		for (int i=0; i<this->nwchanges; ++i){
			delete this->wchanges[i];
		}
		delete [] wchanges;
	}
	if (wordination!=0) delete [] wordination;
}
   		
Neuron * Network::GetNeuronAt(int index) const{
	return &(this->neurons[index]);
}
   		
int Network::GetNeuronNumber() const{
	return this->nneurons;	
}

Neuron * Network::GetTimeDrivenNeuronAt(int index0,int index1) const{
	return this->timedrivenneurons[index0][index1];
}

Neuron * Network::GetTimeDrivenNeuronGPUAt(int index0, int index1) const{
	return 0;
}
   		
int * Network::GetTimeDrivenNeuronNumber() const{
	return this->ntimedrivenneurons;
}


int Network::GetNneutypes() const{
	return this->nneutypes;
}
int * Network::GetTimeDrivenNeuronNumberGPU() const{
	return 0;
}

NeuronModel * Network::GetNeuronModelAt(int index) const{
	return this->neutypes[index];
}


LearningRule * Network::GetLearningRuleAt(int index) const{
	return this->wchanges[index];
}

int Network::GetLearningRuleNumber() const{
	return this->nwchanges;
}

void Network::LoadNet(const char *netfile) throw (EDLUTException){
	FILE *fh;
	long savedcurrentline;
	long Currentline;
	fh=fopen(netfile,"rt");
	if(fh){
		Currentline=1L;
		skip_comments(fh, Currentline);
		if(fscanf(fh,"%i",&(this->nneutypes))==1){
			this->neutypes=(NeuronModel **) new NeuronModel * [this->nneutypes];
			if(this->neutypes){
				int ni;
				for(ni=0;ni<this->nneutypes;ni++){
					this->neutypes[ni]=0;
				}
            	skip_comments(fh, Currentline);
            	if(fscanf(fh,"%i",&(this->nneurons))==1){
            		int tind,nind,nn,outn,monit;
            		NeuronModel * type;
            		char ident[MAXIDSIZE+1];
            		char ident_type[MAXIDSIZE+1];
            		this->neurons=(Neuron *) new Neuron [this->nneurons];

					ntimedrivenneurons= new int [this->nneutypes]();
					int ** time_driven_index = (int **) new int *[this->nneutypes];
					
					for (int z=0; z<this->nneutypes; z++){
						time_driven_index[z]=new int [this->nneurons]();
					}

					int * N_neurons= new int [this->nneutypes]();

            		if(this->neurons){
            			for(tind=0;tind<this->nneurons;tind+=nn){
                     		skip_comments(fh,Currentline);
                     		if(fscanf(fh,"%i",&nn)==1 && fscanf(fh," %"MAXIDSIZEC"[^ ]%*[^ ]",ident_type)==1 && fscanf(fh," %"MAXIDSIZEC"[^ ]%*[^ ]",ident)==1 && fscanf(fh,"%i",&outn)==1 && fscanf(fh,"%i",&monit)==1){
                     			if(tind+nn>this->nneurons){
                     				throw EDLUTFileException(4,7,6,1,Currentline);
                     			}
								int ni;                    
                        		savedcurrentline=Currentline;
                        		type=LoadNetTypes(ident_type, ident, ni);
                        		Currentline=savedcurrentline;

	                    		for(nind=0;nind<nn;nind++){
                        			neurons[nind+tind].InitNeuron(nind+tind, N_neurons[ni], type,(bool) monit, (bool)outn);
								
									//If some neuron is monitored.
									if(monit){
										type->GetVectorNeuronState()->Set_Is_Monitored(true);
									}

									N_neurons[ni]=N_neurons[ni]+1;
									if (type->GetModelType()==TIME_DRIVEN_MODEL_CPU){
										time_driven_index[ni][this->ntimedrivenneurons[ni]] = nind+tind;
										this->ntimedrivenneurons[ni]++;
									}
                        		}
                        	}else{
                        		throw EDLUTFileException(4,8,7,1,Currentline);
                        	}
                     	}

						// Create the time-driven cell array
						timedrivenneurons=(Neuron ***) new Neuron ** [this->nneutypes];
						for (int z=0; z<this->nneutypes; z++){ 
							if (this->ntimedrivenneurons[z]>0){
								this->timedrivenneurons[z]=(Neuron **) new Neuron * [this->ntimedrivenneurons[z]];

								for (int i=0; i<this->ntimedrivenneurons[z]; ++i){
									this->timedrivenneurons[z][i] = &(this->neurons[time_driven_index[z][i]]);
								}							
							}
						}

						// Initialize states. 
						InitializeStates(N_neurons);
						
            		}else{
            			throw EDLUTFileException(4,5,28,0,Currentline);
            		}

					for (int z=0; z<this->nneutypes; z++){
						delete [] time_driven_index[z];
					} 
					delete [] time_driven_index;
					delete [] N_neurons;

            		/////////////////////////////////////////////////////////
            		// Check the number of neuron types
            		for(ni=0;ni<this->nneutypes && this->neutypes[ni]!=0;ni++);

            		if (ni!=this->nneutypes){
            			throw EDLUTException(13,44,20,0);
            		}
            	}else{
            		throw EDLUTFileException(4,9,8,1,Currentline);
            	}
            	
            	skip_comments(fh,Currentline);
        		if(fscanf(fh,"%i",&(this->nwchanges))==1){
        			int wcind;
        			this->wchanges=new LearningRule * [this->nwchanges];
        			if(this->wchanges){
        				for(wcind=0;wcind<this->nwchanges;wcind++){
        					char ident_type[MAXIDSIZE+1];
        					skip_comments(fh,Currentline);
        					string LearningModel;
        					if(fscanf(fh," %"MAXIDSIZEC"[^ ]%*[^ ]",ident_type)==1){
        						if (string(ident_type)==string("ExpAdditiveKernel")){
        							this->wchanges[wcind] = new ExpWeightChange();
        						} else if (string(ident_type)==string("SinAdditiveKernel")){
        							this->wchanges[wcind] = new SinWeightChange();
        						} else if (string(ident_type)==string("CosAdditiveKernel")){
									this->wchanges[wcind] = new CosWeightChange();
        						} else if (string(ident_type)==string("SimetricCosAdditiveKernel")){
									this->wchanges[wcind] = new SimetricCosWeightChange();
        						} else if (string(ident_type)==string("STDP")){
        							this->wchanges[wcind] = new STDPWeightChange();
								} else if (string(ident_type)==string("STDPLS")){
        							this->wchanges[wcind] = new STDPLSWeightChange();
        						} else {
                           			throw EDLUTFileException(4,28,23,1,Currentline);
        						}

        						this->wchanges[wcind]->LoadLearningRule(fh,Currentline);

                       		}else{
                       			throw EDLUTFileException(4,28,23,1,Currentline);
                       		}
        				}
        			}else{
        				throw EDLUTFileException(4,5,4,0,Currentline);
        			}
        		}else{
        			throw EDLUTFileException(4,26,21,1,Currentline);
        		}
int * N_ConectionWithLearning;
if(this->nwchanges>0){          	
N_ConectionWithLearning=new int [this->nwchanges](); 
}
        		skip_comments(fh,Currentline);
        		if(fscanf(fh,"%li",&(this->ninters))==1){
        			int source,nsources,target,ntargets,nreps,wchange,wchange2;
        			float delay,delayinc,maxweight;
        			int type;
        			int iind,sind,tind,rind,posc;
        			this->inters=(Interconnection *) new Interconnection [this->ninters];
        			this->wordination=(Interconnection **) new Interconnection * [this->ninters];
        			if(this->inters && this->wordination){
        				for(iind=0;iind<this->ninters;iind+=nsources*ntargets*nreps){
        					skip_comments(fh,Currentline);
        					if(fscanf(fh,"%i",&source)==1 && fscanf(fh,"%i",&nsources)==1 && fscanf(fh,"%i",&target)==1 && fscanf(fh,"%i",&ntargets)==1 && fscanf(fh,"%i",&nreps)==1 && fscanf(fh,"%f",&delay)==1 && fscanf(fh,"%f",&delayinc)==1 && fscanf(fh,"%i",&type)==1 && fscanf(fh,"%f",&maxweight)==1 && fscanf(fh,"%i",&wchange)==1){
								wchange2=-1;
								if(wchange>=0){
									if(is_end_line(fh,Currentline)==false){
										if(fscanf(fh,"%i",&wchange2)==1){
											if(wchange2>= this->nwchanges){
  												throw EDLUTFileException(4,29,24,1,Currentline);
											}
										}else{
											throw EDLUTFileException(4,12,11,1,Currentline);
										}
									}
								}
								if(iind+nsources*ntargets*nreps>this->ninters){
        							throw EDLUTFileException(4,10,9,1,Currentline);
        						}else{
        							if(source+nreps*nsources>this->nneurons || target+nreps*ntargets>this->nneurons){
  										throw EDLUTFileException(4,11,10,1,Currentline);
  									}else{
  										if(wchange >= this->nwchanges){
  											throw EDLUTFileException(4,29,24,1,Currentline);
  										}
        							}
        						}
        						
        						for(rind=0;rind<nreps;rind++){
        							for(sind=0;sind<nsources;sind++){
        								for(tind=0;tind<ntargets;tind++){
        									posc=iind+rind*nsources*ntargets+sind*ntargets+tind;
        									this->inters[posc].SetIndex(posc);
        									this->inters[posc].SetSource(&(this->neurons[source+rind*nsources+sind]));
        									this->inters[posc].SetTarget(&(this->neurons[target+rind*ntargets+tind]));
        									//Net.inters[posc].target=target+rind*nsources+tind;  // other kind of neuron arrangement
        									this->inters[posc].SetDelay(delay+delayinc*tind);
        									this->inters[posc].SetType(type);
        									//this->inters[posc].nextincon=&(this->inters[posc]);       // temporaly used as weight index
        									this->inters[posc].SetWeight(maxweight);   //TODO: Use max interconnection conductance
        									this->inters[posc].SetMaxWeight(maxweight);

											this->inters[posc].SetWeightChange_withPost(0);
											this->inters[posc].SetWeightChange_withoutPost(0);
											if(wchange >= 0){
												//Set the new learning rule
												if(wchanges[wchange]->ImplementPostSynaptic()==true){
													this->inters[posc].SetWeightChange_withPost(this->wchanges[wchange]);
												}else{
													this->inters[posc].SetWeightChange_withoutPost(this->wchanges[wchange]);
												}
												N_ConectionWithLearning[wchange]++;
											}

											if(wchange2>= 0){
												//Set the new learning rule
												if(wchanges[wchange2]->ImplementPostSynaptic()==true){
													this->inters[posc].SetWeightChange_withPost(this->wchanges[wchange2]);
												}else{
													this->inters[posc].SetWeightChange_withoutPost(this->wchanges[wchange2]);
												}
												N_ConectionWithLearning[wchange2]++;
											}
                                		}
        							}
        						}
        					}else{
        						throw EDLUTFileException(4,12,11,1,Currentline);
        					}
        				}
for(int t=0; t<this->nwchanges; t++){
	if(N_ConectionWithLearning[t]>0){
		this->wchanges[t]->InitializeConnectionState(N_ConectionWithLearning[t]);
	}
}
if(this->nwchanges>0){
delete [] N_ConectionWithLearning;
}
        				
						FindOutConnections();
                    	SetWeightOrdination(); // must be before find_in_c() and after find_out_c()
                    	FindInConnections();
                    }else{
        				throw EDLUTFileException(4,5,28,0,Currentline);
        			}
        		}else{
        			throw EDLUTFileException(4,13,12,1,Currentline);
        		}
            }else{
            	throw EDLUTFileException(4,5,4,0,Currentline);
			}
		}else{
			throw EDLUTFileException(4,6,5,1,Currentline);
		}
		
		fclose(fh);
	}else{
		throw EDLUTFileException(4,14,13,0,Currentline);
	}
	
	return;
}

void Network::LoadWeights(const char *wfile) throw (EDLUTFileException){
	FILE *fh;
	int connind;
	long Currentline;
	fh=fopen(wfile,"rt");
	if(fh){
		int nweights,weind;
		float weight;
		Currentline=1L;
		skip_comments(fh,Currentline);
		for(connind=0;connind<this->ninters;connind+=nweights){
			skip_comments(fh, Currentline);
			if(fscanf(fh,"%i",&nweights)==1 && fscanf(fh,"%f",&weight)==1){
				if(nweights < 0 || nweights + connind > this->ninters){
					throw EDLUTFileException(11,32,26,1,Currentline);
				}
				
				if(nweights == 0){
					nweights=this->ninters-connind;
				}
				
				for(weind=0;weind<nweights;weind++){
					Interconnection * Connection = this->wordination[connind+weind];
					Connection->SetWeight(((weight < 0.0)?rand()*Connection->GetMaxWeight()/RAND_MAX:((weight > Connection->GetMaxWeight())?Connection->GetMaxWeight():weight)));
				}
			}else{
				throw EDLUTFileException(11,31,25,1,Currentline);
			}
		}
		fclose(fh);
	}else{
		throw EDLUTFileException(11,30,13,0,Currentline);
	}
	
}

void Network::SaveWeights(const char *wfile) throw (EDLUTException){
	FILE *fh;
	int connind;
	fh=fopen(wfile,"wt");
	if(fh){
		float weight,antweight;
		int nantw;
		nantw=0;
		antweight=0.0;
		weight=0.0; // just to avoid compiler warning messages
		
		// Write the number of weights
		//if(fprintf(fh,"%li\n",this->ninters) <= 0){
		//	throw EDLUTException(12,33,4,0);
		//}
					
		for(connind=0;connind<=this->ninters;connind++){
			if(connind < this->ninters){
				weight=this->wordination[connind]->GetWeight();
			}
			
			if(antweight != weight || connind == this->ninters){
				if(nantw > 0){
					if(fprintf(fh,"%i %g\n",nantw,antweight) <= 0){
						throw EDLUTException(12,33,4,0);
					}
				}
				
				antweight=weight;
				nantw=1;
			}else{
				nantw++;
			}
		}
		
		// fprintf(fh,"// end of written data\n");
		
		fclose(fh);
	}else{
		throw EDLUTException(12,30,27,0);
	}
	
}

ostream & Network::PrintInfo(ostream & out) {
	int ind;

	out << "- Neuron types:" << endl;

	for(ind=0;ind<this->nneutypes;ind++){
		out << "\tType: " << ind << endl;

		this->neutypes[ind]->PrintInfo(out);
	}
   
	out << "- Neurons:" << endl;
   	
   	for(ind=0;ind<this->nneurons;ind++){
		this->neurons[ind].PrintInfo(out);
	}

   	out << "- Weight change types:" << endl;

	for(ind=0;ind<this->nwchanges;ind++){
		out << "\tChange: " << ind << endl;
		this->wchanges[ind]->PrintInfo(out);
	}

	out << "- Interconnections:" << endl;

	for(ind=0; ind<this->ninters; ind++){
		out << "\tConnection: " << ind << endl;
		this->inters[ind].PrintInfo(out);
	}
	
	return out;
}


