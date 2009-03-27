/***************************************************************************
 *                           Network.cpp                                   *
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

#include "../../include/spike/Network.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/NeuronType.h"
#include "../../include/spike/MultiplicativeWeightChange.h"
#include "../../include/spike/AdditiveWeightChange.h"

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
		for (int ninter=0;ninter<ninters;ninter++){
			inters[ninter].GetSource()->AddOutputConnection(inters+ninter);
		}		
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
	if(ninters>0){
		for (int ninter=0;ninter<ninters;ninter++){
			inters[ninter].GetTarget()->AddInputConnection(inters+ninter);
		}
	}
}

NeuronType * Network::LoadNetTypes(char *neutype) throw (EDLUTException){
	int ni;
   	NeuronType * type;
   	
   	for(ni=0;ni<nneutypes && strcmp(neutypes[ni].GetId(),neutype);ni++);
   
	if(ni<nneutypes){
		type=neutypes+ni;
	}else{
		for(ni=0;ni<nneutypes && neutypes[ni].GetId()[0] != '\0';ni++);
		
		if(ni<nneutypes){
			neutypes[ni].LoadNeuronType(neutype);
			type=neutypes+ni;		
		}else{
			throw EDLUTException(13,44,20,0);
		}
	}	
	return(type);
}

void Network::InitNetPredictions(EventQueue * Queue){
	int nneu;
	for(nneu=0;nneu<nneurons;nneu++)
		(neurons+nneu)->InitNeuronPrediction(Queue);
}

Network::Network(const char * netfile, const char * wfile, EventQueue * Queue) throw (EDLUTException): inters(0), ninters(0), neutypes(0), nneutypes(0), neurons(0), nneurons(0), wchanges(0), nwchanges(0), wordination(0){
	this->LoadNet(netfile);	
	this->LoadWeights(wfile);
	this->LoadNeuronTypeTables();
	this->InitNetPredictions(Queue);	
}
   		
Network::~Network(){
	if (inters!=0) delete [] inters;
   	if (neutypes!=0) delete [] neutypes;
	if (neurons!=0) delete [] neurons;
	if (wchanges!=0) delete [] wchanges;
	if (wordination!=0) delete [] wordination;
}
   		
Neuron * Network::GetNeuronAt(int index) const{
	return &(this->neurons[index]);
}
   		
int Network::GetNeuronNumber() const{
	return this->nneurons;	
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
			this->neutypes=(NeuronType *) new NeuronType [this->nneutypes];
			if(this->neutypes){
				int ni;
				for(ni=0;ni<this->nneutypes;ni++){
					this->neutypes[ni].ClearNeuronType();
				}
            	skip_comments(fh, Currentline);
            	if(fscanf(fh,"%i",&(this->nneurons))==1){
            		int tind,nind,nn,outn,monit;
            		NeuronType * type;
            		char ident[MAXIDSIZE+1];
            		this->neurons=(Neuron *) new Neuron [this->nneurons];
            		if(this->neurons){
            			for(tind=0;tind<this->nneurons;tind+=nn){
                     		skip_comments(fh,Currentline);
                     		if(fscanf(fh,"%i",&nn)==1 && fscanf(fh," %"MAXIDSIZEC"[^ ]%*[^ ]",ident)==1 && fscanf(fh,"%i",&outn)==1 && fscanf(fh,"%i",&monit)==1){
                     			if(tind+nn>this->nneurons){
                     				throw EDLUTFileException(4,7,6,1,Currentline);
                     				break;
                     			}
                        
                        		savedcurrentline=Currentline;
                        		type=LoadNetTypes(ident);
                        		Currentline=savedcurrentline;
                        
                        		for(nind=0;nind<nn;nind++){
                        			neurons[nind+tind].InitNeuron(nind+tind, type,(bool) monit, (bool)outn);
                        		}
                        	}else{
                        		throw EDLUTFileException(4,8,7,1,Currentline);
                        		break;
                        	}
                     	}
            		}else{
            			throw EDLUTFileException(4,5,28,0,Currentline);
            		}
            	}else{
            		throw EDLUTFileException(4,9,8,1,Currentline);
            	}
            	
            	skip_comments(fh,Currentline);
        		if(fscanf(fh,"%i",&(this->nwchanges))==1){
        			float maxpos,a1pre,a2prepre;
        			int trigger;
        			int multiplicative;
        			int wcind;
        			this->wchanges=new WeightChange * [this->nwchanges];
        			if(this->wchanges){
        				for(wcind=0;wcind<this->nwchanges;wcind++){
        					int indexp;
        					static float explpar[]={30.1873,60.3172,5.9962};
        					static float expcpar[]={-5.2410,3.1015,2.2705};
        					skip_comments(fh,Currentline);
        					if(fscanf(fh,"%i",&trigger)==1 && fscanf(fh,"%f",&maxpos)==1 && fscanf(fh,"%f",&a1pre)==1 && fscanf(fh,"%f",&a2prepre)==1 && fscanf(fh,"%i",&multiplicative)==1){
        						if(a1pre < -1.0 || a1pre > 1.0){
        							throw EDLUTFileException(4,27,22,1,Currentline);
        							break;
        						}
        						if (multiplicative==1){
        							this->wchanges[wcind] = new MultiplicativeWeightChange();
        						}else{
        							this->wchanges[wcind] = new AdditiveWeightChange();
        						}
        						this->wchanges[wcind]->SetTrigger(trigger);
                       			this->wchanges[wcind]->SetMaxPos(maxpos);
                       			this->wchanges[wcind]->SetNumExps(3);
                       			for(indexp=0;indexp<this->wchanges[wcind]->GetNumExps();indexp++){
                       				this->wchanges[wcind]->SetLparAt(indexp,(maxpos == 0)?0:(0.1/maxpos)*explpar[indexp]);
                       				this->wchanges[wcind]->SetCparAt(indexp,expcpar[indexp]);
                       			}
                       			
                       			this->wchanges[wcind]->SetA1Pre(a1pre);
                       			this->wchanges[wcind]->SetA2PrePre(a2prepre);
                       		}else{
                       			throw EDLUTFileException(4,28,23,1,Currentline);
                       			break;
                       		}
        				}
        			}else{
        				throw EDLUTFileException(4,5,4,0,Currentline);
        			}
        		}else{
        			throw EDLUTFileException(4,26,21,1,Currentline);
        		}
            	
            	
        		skip_comments(fh,Currentline);
        		if(fscanf(fh,"%i",&(this->ninters))==1){
        			int source,nsources,target,ntargets,nreps,wchange;
        			float delay,delayinc,maxweight;
        			int type;
        			int iind,sind,tind,rind,posc;
        			this->inters=(Interconnection *) new Interconnection [this->ninters];
        			this->wordination=(Interconnection **) new Interconnection * [this->ninters];
        			if(this->inters && this->wordination){
        				for(iind=0;iind<this->ninters;iind+=nsources*ntargets*nreps){
        					skip_comments(fh,Currentline);
        					if(fscanf(fh,"%i",&source)==1 && fscanf(fh,"%i",&nsources)==1 && fscanf(fh,"%i",&target)==1 && fscanf(fh,"%i",&ntargets)==1 && fscanf(fh,"%i",&nreps)==1 && fscanf(fh,"%f",&delay)==1 && fscanf(fh,"%f",&delayinc)==1 && fscanf(fh,"%i",&type)==1 && fscanf(fh,"%f",&maxweight)==1 && fscanf(fh,"%i",&wchange)==1){
        						if(iind+nsources*ntargets*nreps>this->ninters){
        							throw EDLUTFileException(4,10,9,1,Currentline);
        							break;
        						}else{
        							if(source+nreps*nsources>this->nneurons || target+nreps*ntargets>this->nneurons){
  										throw EDLUTFileException(4,11,10,1,Currentline);
  										break;
  									}else{
  										if(wchange >= this->nwchanges){
  											throw EDLUTFileException(4,29,24,1,Currentline);
  											break;
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
        									if(wchange >= 0){
        										this->inters[posc].ClearActivity();
        										this->inters[posc].SetWeightChange(this->wchanges[wchange]);
        									}
                                
                                			this->inters[posc].SetLastSpikeTime(0); // -1.0/0.0; // -Infinite not needed if last activity=0
                                		}
        							}
        						}
        					}else{
        						throw EDLUTFileException(4,12,11,1,Currentline);
        						break;
        					}
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
					break;
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
				break;
			}
		}
		fclose(fh);
	}else{
		throw EDLUTFileException(11,30,13,0,Currentline);
	}
	
}

void Network::SaveWeights(const char *wfile) throw (EDLUTException){
	FILE *fh;
	int connind,ret;
	long Currentline;
	fh=fopen(wfile,"wt");
	if(fh){
		float weight,antweight;
		int nantw;
		Currentline=1L;
		ret=0;
		nantw=0;
		antweight=0.0;
		weight=0.0; // just to avoid compiler warning messages
		
		// Write the number of weights
		if(fprintf(fh,"%li\n",this->ninters) <= 0){
			throw EDLUTException(12,33,4,0);
		}
					
		for(connind=0;connind<=this->ninters;connind++){
			if(connind < this->ninters){
				weight=this->wordination[connind]->GetWeight();
			}
			
			if(antweight != weight || connind == this->ninters){
				if(nantw > 0){
					if(fprintf(fh,"%i %g\n",nantw,antweight) <= 0){
						throw EDLUTException(12,33,4,0);
						break;
					}
				}
				
				antweight=weight;
				nantw=1;
			}else{
				nantw++;
			}
		}
		
		fprintf(fh,"// end of written data\n");
		
		fclose(fh);
	}else{
		throw EDLUTException(12,30,27,0);
	}
	
}

void Network::LoadNeuronTypeTables() throw (EDLUTException){
	int ntype;
	for(ntype=0;ntype<this->nneutypes;ntype++){
		this->neutypes[ntype].LoadTables();
	}
}

ostream & Network::GetNetInfo(ostream & out) const{
	int ind,ind2,wneu=0;
	out << "Neuron types:" << endl;
	for(ind=0;ind<this->nneutypes;ind++){
		out << "- Type: " << ind << " state vars: " << this->neutypes[ind].GetStateVarsNumber() << " tables: " << this->neutypes[ind].GetTableNumber() << endl;
	}
   
	if(ind < this->nneutypes){
		out << "..." << this->nneutypes-1 << endl;
	}
   
   	out << "Neurons:" << endl;
   	
   	for(ind=0;ind<this->nneurons;ind++){
		out << "- Neuron: " << ind << " typ: " << this->neurons[ind].GetNeuronType()->GetId() << " v0: " << this->neurons[ind].GetStateVarAt(1) << " v1: " << this->neurons[ind].GetStateVarAt(2) << " lastup: " << this->neurons[ind].GetLastUpdate() << " predsp: " << this->neurons[ind].GetPredictedSpike() << " " << this->neurons[ind].GetOutputNumber() << " " << (this->neurons[ind].IsMonitored()?"o":"") << endl;
		wneu=0;
      	for(ind2=0;ind2<this->ninters && this->inters[ind2].GetTarget() != this->neurons+ind && this->inters[ind2].GetSource() != this->neurons+ind ;ind2++);
		if(ind2 == this->ninters)
        	wneu=1;
	}
   
	if(ind < this->nneurons){
		out << "..." << this->nneurons-1 << endl;
	}
   
	if(wneu)
		out << "Warning: There are neurons without connection" << endl;

	out << "Weight change types:" << endl;
	for(ind=0;ind<this->nwchanges;ind++){
		out << "- Change: " << ind << " trigger: " << this->wchanges[ind]->GetTrigger() << " max pos: " << this->wchanges[ind]->GetMaxPos() << " alpha: " << this->wchanges[ind]->GetA1Pre() << " beta: " << this->wchanges[ind]->GetA2PrePre() << endl;
	}

	if(ind < this->nwchanges){
		out << "..." << this->nwchanges-1 << endl;
	}
	
	out << "Interconnections:" << endl;

	for(ind=0;ind<this->ninters; ind++){
		out << "- Interc: " << ind << " source: " << this->inters[ind].GetSource()->GetIndex() << " target: " << this->inters[ind].GetTarget()->GetIndex() << " delay: " << this->inters[ind].GetDelay() << " type: " << ((this->inters[ind].GetType()==1)?"E":((this->inters[ind].GetType()==0)?"I":"C")) << " weight: " << this->inters[ind].GetWeight() << endl;
	}

	if(ind < this->ninters){
		out << "..." << this->ninters-1 << endl;
	}
	
	return out;
}


