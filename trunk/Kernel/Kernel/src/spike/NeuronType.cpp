/***************************************************************************
 *                           NeuronType.cpp                                *
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

#include "../../include/spike/NeuronType.h"

#include "../../include/spike/NeuronModelTable.h"

#include "../../include/simulation/Configuration.h"
#include "../../include/simulation/Utils.h"

char * NeuronType::GetId(){
	return this->ident;	
}
   		
int NeuronType::GetStateVarsNumber() const{
	return this->nstatevars;
}
   		
int NeuronType::GetTimeDependentStateVarsNumber() const{
	return this->ntstatevars;
}
   		
int NeuronType::GetStateVarAt(int index) const{
	return this->statevarorder[index];
}
   		
int NeuronType::GetStateVarTableAt(int index) const{
	return this->statevartables[index];
}
   		
float NeuronType::GetInitValueAt(int index) const{
	return this->initvalues[index];
}
   		
int NeuronType::GetFiringTable() const{
	return this->firingtable;
}
   		
int NeuronType::GetFiringEndTable() const{
	return this->firingendtable;
}
   		
int NeuronType::GetSynapticVarsNumber() const{
	return this->nsynapticvars;
}
   		
int NeuronType::GetSynapticVarsAt(int index) const{
	return this->synapticvars[index];
}
   		
NeuronModelTable * NeuronType::GetTableAt(int index){
	return this->tables+index;
}
   		
int NeuronType::GetTableNumber() const{
	return this->ntables;
}

void NeuronType::ClearNeuronType(){
	this->ident[0]='\0';
	this->ntables=0;
	this->nstatevars=0;
}

void NeuronType::TypeInfo(){
    unsigned long idim,itab;
    printf("Ident: %s\n",this->ident);
    for(itab=0;itab<this->ntables;itab++){
        printf("%lu %i(%+i)   ",this->tables[itab].GetDimensionNumber(),this->tables[itab].GetInterpolation(),this->tables[itab].GetFirstInterpolation());
        for(idim=0;idim<this->tables[itab].GetDimensionNumber();idim++){
        	printf("%i %i(%i)  ",this->tables[itab].GetDimensionAt(idim)->statevar,this->tables[itab].GetDimensionAt(idim)->interp,this->tables[itab].GetDimensionAt(idim)->nextintdim);
        }
      	printf("\n");
     }
  }

			
void NeuronType::LoadNeuronType(char * neutype) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	char neufile[MAXIDSIZE+5];
	//type = neutypes+ni;
	strcpy(this->ident,neutype);
	strcpy(neufile,neutype);
	strcat(neufile,".cfg");
	fh=fopen(neufile,"rt");
	if(fh){
		Currentline=1L;
		skip_comments(fh,Currentline);
		if(fscanf(fh,"%i",&this->nstatevars)==1){
			unsigned int nv;
			if(this->nstatevars < MAXSTATEVARS){
				skip_comments(fh,Currentline);
				for(nv=0;nv<this->nstatevars;nv++){
					if(fscanf(fh,"%i",&this->statevartables[nv])!=1){
						throw EDLUTFileException(13,41,3,1,Currentline);
					}
				}
				
				skip_comments(fh,Currentline);
          
          		for(nv=0;nv<this->nstatevars;nv++){
          			if(fscanf(fh,"%f",&this->initvalues[nv])!=1){
          				throw EDLUTFileException(13,42,3,1,Currentline);
          			}
          		}
          	} else{
          		throw EDLUTFileException(13,4,29,1,Currentline);
			}
       
   			skip_comments(fh,Currentline);
   			if(fscanf(fh,"%i",&this->firingtable)==1){
   				skip_comments(fh,Currentline);
   				if(fscanf(fh,"%i",&this->firingendtable)==1){
   					skip_comments(fh,Currentline);
   					if(fscanf(fh,"%i",&this->nsynapticvars)==1){
               			skip_comments(fh,Currentline);
               			for(nv=0;nv<this->nsynapticvars;nv++){
                  			if(fscanf(fh,"%i",&this->synapticvars[nv])!=1){
                  				throw EDLUTFileException(13,40,3,1,Currentline);
                  			}
                  		}
                  		
              			skip_comments(fh,Currentline);
              			if(fscanf(fh,"%i",&this->ntables)==1){
              				unsigned int nt;
              				int tdeptables[MAXSTATEVARS];
              				int tstatevarpos,ntstatevarpos,ctablen;
              				
              				for(nt=0;nt<this->ntables;nt++){
              					this->tables[nt].LoadTableDescription(fh, Currentline);	
                   			}
                 
                 			this->ntstatevars=0;
                 			for(nt=0;nt<this->nstatevars;nt++){
         						ctablen=this->statevartables[nt];
            					for(nv=0;nv<this->tables[ctablen].GetDimensionNumber() && this->tables[ctablen].GetDimensionAt(nv)->statevar != 0;nv++);
            					if(nv<this->tables[ctablen].GetDimensionNumber()){
            						tdeptables[nt]=1;
            						this->ntstatevars++;
            					}else{
               						tdeptables[nt]=0;
            					}
            				}
         
         					tstatevarpos=0;
         					ntstatevarpos=this->ntstatevars; // we place non-t-depentent variables in the end, so that they are evaluated afterwards
         					for(nt=0;nt<this->nstatevars;nt++){
            					this->statevarorder[(tdeptables[nt])?tstatevarpos++:ntstatevarpos++]=nt;
         					}
              			}else{
         					throw EDLUTFileException(13,37,3,1,Currentline);
      					}
      				}else{
       					throw EDLUTFileException(13,36,3,1,Currentline);
          			}
				}else{
    				throw EDLUTFileException(13,43,3,1,Currentline);
          		}
			}else{
 				throw EDLUTFileException(13,35,3,1,Currentline);
			}
		}else{
			throw EDLUTFileException(13,34,3,1,Currentline);
		}
	}else{
		throw EDLUTFileException(13,25,13,0,Currentline);
	}
}	
		
		

void NeuronType::LoadTables() throw (EDLUTException){
	FILE *fd;
	unsigned int i;
	NeuronModelTable * tab;
	char tablefile[MAXIDSIZE+5];
	strcpy(tablefile,this->ident);
	strcat(tablefile,".dat");
	fd=fopen(tablefile,"rb");
	if(fd){
		for(i=0;i<this->ntables;i++){
			tab=&this->tables[i];
			tab->LoadTable(fd);
			if(tab->GetDimensionNumber() != this->tables[i].GetDimensionNumber()){
				//char msgbuf[160];
				//sprintf(msgbuf,"The table %i of file %s has a diferent number of dimensions from the one specified in the configuration file",i,type->ident);
				//show_error(msgbuf);
			}
		}
		fclose(fd);
	}else{
		throw EDLUTException(10,24,13,0);
	}
}

float NeuronType::TableAccess(int ntab, float *statevars){
	return this->tables[ntab].TableAccess(statevars);
}


