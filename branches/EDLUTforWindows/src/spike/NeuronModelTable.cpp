/***************************************************************************
 *                           NeuronModelTable.cpp                          *
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

#include "../../include/spike/NeuronModelTable.h"

#include "../../include/simulation/Utils.h"

#include <cfloat>

NeuronModelTable::TableDimension::TableDimension(): size(0), coord(0), vindex(0), voffset(0), vscale(0), vfirst(0), statevar(0), interp(0), nextintdim(0) {
	
}
   				
NeuronModelTable::TableDimension::~TableDimension(){
	if (coord!=0) delete [] coord;
   			
    if (vindex!=0) delete [] vindex;
    
    if (voffset!=0) delete [] voffset;
}

NeuronModelTable::NeuronModelTable(): elems(0), ndims(0), nelems(0), interp(0), firstintdim(0){
	
}
  		
NeuronModelTable::~NeuronModelTable(){
	if (elems!=0) delete [] (float *) elems;	
}

void NeuronModelTable::GenerateVirtualCoordinates() throw (EDLUTException){
	unsigned long idim,icoord;
	float minsca,sca,first,last;
	unsigned int size;
	for(idim=0;idim<this->ndims;idim++){ // for each dimension of the table
		minsca=FLT_MAX;  // search for the minimum coordinate increment
		for(icoord=1;icoord < this->dims[idim].size;icoord++){
			sca=(this->dims[idim].coord[icoord] - this->dims[idim].coord[icoord-1]);
			if(sca < minsca && sca != 0.0){
				minsca=sca;
			}
		}
		
		first=this->dims[idim].coord[0];
		last=this->dims[idim].coord[this->dims[idim].size-1];
		this->dims[idim].vfirst=first;
		this->dims[idim].vscale=minsca;
		size=unsigned((last-first)/minsca+2);
		this->dims[idim].vindex=(int *) new int [size];
		this->dims[idim].voffset=(float *) new float [size];
		if(this->dims[idim].vindex && this->dims[idim].voffset){  // create virtual coordinates
			unsigned int ivind,ipos;
			float coffset;
			ipos=0;          // dimension coordinates must grow monotonously
			for(ivind=0;ivind<size;ivind++){
				if(this->dims[idim].interp){  // interpolated dimension
					if(ipos+2 < this->dims[idim].size && this->dims[idim].coord[ipos+1] < ivind*minsca+first){
						ipos++;
					}
					
					if(ipos+1 <  this->dims[idim].size){
						coffset=1.0-(this->dims[idim].coord[ipos+1] - (ivind*minsca+first))/minsca;
						
						if(coffset < 0.0){
							coffset=0.0;
						}
					}else{
						coffset=0.0;
					}
				}else{  // non interpolated dimension
					if(ipos+1 < this->dims[idim].size && (this->dims[idim].coord[ipos]+this->dims[idim].coord[ipos+1])/2 < ivind*minsca+first){
						ipos++;
					}
					
					coffset=0.0;
					
					if(ipos+1 < this->dims[idim].size){
						coffset=(ivind*minsca+first)/minsca + 0.5 - (this->dims[idim].coord[ipos]+this->dims[idim].coord[ipos+1])/2/minsca;
						if(coffset < 0.0){
							coffset=0.0;
						}
					}
					
					if(ipos > 0 && coffset == 0.0){
						coffset=(ivind*minsca+first)/minsca - 0.5 - (this->dims[idim].coord[ipos]+this->dims[idim].coord[ipos-1])/2/minsca;
						
						if(coffset > 0.0){
							coffset=0.0;
						}
					}
				}
				
				this->dims[idim].voffset[ivind]=coffset;
				this->dims[idim].vindex[ivind]=ipos;
			}
		}else{
			delete [] this->dims[idim].vindex;
			this->dims[idim].vindex=0;
			delete [] this->dims[idim].voffset;
			this->dims[idim].voffset=0;
			break;
		}
	}
	
	if(idim != this->ndims){
		throw EDLUTException(7,5,4,0);
	}
	
}

void NeuronModelTable::TableInfo()
{
   unsigned long idim;
   printf("Number of elements: %lu\tNumber of dimensions: %lu\n",this->nelems,this->ndims);
   for(idim=0;idim < this->ndims;idim++){
      printf("Dimension %lu: size %lu vsize %i vscale %g coords: ",idim,this->dims[idim].size,(int)((this->dims[idim].coord[this->dims[idim].size-1]-this->dims[idim].coord[0])/this->dims[idim].vscale+2),this->dims[idim].vscale);
      if(this->dims[idim].size>0){
         printf("[%g, %g]",this->dims[idim].coord[0],this->dims[idim].coord[this->dims[idim].size-1]);
      }
      printf("\n");
   }
}


void NeuronModelTable::LoadTable(FILE *fd) throw (EDLUTException){
	void **elems;
	unsigned long idim,totsize,vecsize;
	uint64_t nelems;
	 
	if(fread(&(nelems),sizeof(uint64_t),1,fd)==1){
		this->nelems = nelems;
		if(this->nelems>0 && (uint64_t)(this->nelems)==nelems){
			uint64_t ndims;
			if(fread(&ndims,sizeof(uint64_t),1,fd)==1){
            	this->ndims=ndims;

				vecsize=1L;
				totsize=0L;
				for(idim=0;idim < this->ndims;idim++){
					uint64_t dsize;
               		if(fread(&dsize,sizeof(uint64_t),1,fd)==1){
                  		this->dims[idim].size=dsize;
						vecsize*=this->dims[idim].size;
						totsize+=vecsize;
						this->dims[idim].coord=(float *) new float [this->dims[idim].size];
						if(this->dims[idim].coord){
							if(fread(this->dims[idim].coord,sizeof(float),this->dims[idim].size,fd)!=this->dims[idim].size){
								throw EDLUTException(9,21,19,0);
								break;
							}
						}else{
							throw EDLUTException(9,5,4,0);
							break;
						}
					}else{
						throw EDLUTException(9,21,19,0);
						break;
					}
				}
				
				if(idim==this->ndims){
					if(this->nelems != vecsize){
						char msgbuf[80];
						sprintf(msgbuf,"Inconsisten table (%lu elements expected)",vecsize);
						//show_info(msgbuf);
					}
					
					elems=(void **) malloc((totsize-this->nelems)*sizeof(void *)+this->nelems*sizeof(float));
					if(elems){
						unsigned long i;
						if(fread(elems+totsize-this->nelems,sizeof(float),this->nelems,fd) == this->nelems){
							vecsize=1L;
							totsize=0L;
							for(idim=0;idim < this->ndims-1;idim++){
								long relpos;
                        		void *basepos;
                        		vecsize*=this->dims[idim].size;
								for(i=0;i < vecsize;i++){
									relpos=i*this->dims[idim+1].size;
                           			basepos=elems+totsize+vecsize;
                           			*(elems+totsize+i)=(idim+1 < this->ndims-1)?(void **)basepos+relpos:(void **)((float *)basepos+relpos);
								}
								totsize+=vecsize;
							}
							
							this->elems=elems;
							GenerateVirtualCoordinates();
						}else{
							throw EDLUTException(9,21,19,0);
						}
					}else{
						throw EDLUTException(9,5,4,0);
					}
				}
			}else{
            	throw EDLUTException(9,21,19,0);
			}
		}else{
			(nelems>0)?throw EDLUTException(9,45,19,0):throw EDLUTException(9,23,19,0);
		}
	}else{
		throw EDLUTException(9,22,19,0);
	}
}

void NeuronModelTable::LoadTableDescription(FILE *fh, long & Currentline) throw (EDLUTFileException){
	int previntdim;
	int nv;
	                          				
    this->elems=0;
	this->interp=0;
	skip_comments(fh,Currentline);
	
	if(fscanf(fh,"%li",&this->ndims)==1){
		previntdim=-1;
		this->firstintdim=-1;
		for(nv=this->ndims-1;nv>=0;nv--){
			this->dims[nv].coord=0;
			this->dims[nv].vindex=0;
			this->dims[nv].voffset=0;
			if(fscanf(fh,"%i",&this->dims[nv].statevar)!=1 || 
				fscanf(fh,"%i",&this->dims[nv].interp)!=1){
				throw EDLUTFileException(13,39,3,1,Currentline);
			}else{
				if(this->dims[nv].interp){
					if(this->firstintdim==-1){
						this->firstintdim=nv;
					}else{
						this->dims[previntdim].nextintdim=previntdim-nv;
					}
					previntdim=nv;
					this->interp=this->dims[nv].interp;
   				}else{
    				this->dims[nv].nextintdim=1; // useless value
				}
			}
		}
			
		if(previntdim != -1){
			this->dims[previntdim].nextintdim=previntdim-nv;
		}
	}else{
		throw EDLUTFileException(13,38,3,1,Currentline);
	}
}


const NeuronModelTable::TableDimension * NeuronModelTable::GetDimensionAt(int index) const{
	return this->dims+index;	
}
  		
void NeuronModelTable::SetDimensionAt(int index, NeuronModelTable::TableDimension Dimension){
	this->dims[index]=Dimension;	
}
  		
unsigned long NeuronModelTable::GetDimensionNumber() const{
	return this->ndims;
}
  		
float NeuronModelTable::GetElementAt(int index) const{
	return ((float *) this->elems)[index];
}
  		
void NeuronModelTable::SetElementAt(int index, float Element){
	((float *) this->elems)[index] = Element;
}  		

unsigned long NeuronModelTable::GetElementsNumber() const{
	return this->nelems;
}

int NeuronModelTable::GetInterpolation() const{
	return this->interp;
}
  		
int NeuronModelTable::GetFirstInterpolation() const{
	return this->firstintdim;
}

float NeuronModelTable::TableAccessDirect(float statevars[MAXSTATEVARS]){
	unsigned int idim,tind;
	float elem;
	void **cpointer;
	NeuronModelTable *tab;
	NeuronModelTable::TableDimension *dim;
	tab=this;
	cpointer=(void **)tab->elems;
	for(idim=0;idim<tab->ndims-1;idim++){
		dim=&tab->dims[idim];
		tind=table_indcomp2(dim,statevars[dim->statevar]);
		cpointer=(void **)cpointer[tind];
	}
	dim=&tab->dims[idim];
	tind=table_indcomp2(dim,statevars[dim->statevar]);
	elem=((float *)cpointer)[tind];
	return(elem);
}

// Bilineal interpolation
float NeuronModelTable::TableAccessInterpBi(float statevars[MAXSTATEVARS]){
	unsigned int idim;
	float elem,coord,*coords;
	NeuronModelTable *tab;
	NeuronModelTable::TableDimension *dim;
	int intstate[MAXSTATEVARS]={0};
	float subints[MAXSTATEVARS];
	float coeints[MAXSTATEVARS];
	void  **dpointers[MAXSTATEVARS];
	int tableinds[MAXSTATEVARS];

	tab=this;
	dpointers[0]=(void **)tab->elems;

	for(idim=0;idim<tab->ndims;idim++){
		dim=&tab->dims[idim];
		coord=statevars[dim->statevar];
		if(dim->interp){
			coords=dim->coord;
			if(coord>last_coord(dim)){
				tableinds[idim]=dim->size-2;
				coeints[idim]=1;
			}else{
				if(coord<dim->vfirst){
					tableinds[idim]=0;
					coeints[idim]=0;
				}else{
					tableinds[idim]=table_ind_int(dim,coord);
					coeints[idim]=((coord-coords[tableinds[idim]])/(coords[tableinds[idim]+1]-coords[tableinds[idim]]));
				}
			}
		} else {
			tableinds[idim]=table_indcomp2(dim,coord);
		}
	}
	
	idim=0;
	do{
		for(;idim<tab->ndims-1;idim++){
			dpointers[idim+1]=(void **)dpointers[idim][tableinds[idim]+intstate[idim]];
		}
		
		elem=((float *)dpointers[idim])[tableinds[idim]+intstate[idim]];

		for(idim=tab->firstintdim;idim>=0;idim-=tab->dims[idim].nextintdim){
			intstate[idim]=!intstate[idim];
			if(intstate[idim]){
				subints[idim]=elem;
				break;
			}else{
				elem=subints[idim]=subints[idim]+(elem-subints[idim])*coeints[idim];
			}
		}
	} while(idim>=0);
   
	return(elem);
}

// Lineal interpolation
float NeuronModelTable::TableAccessInterpLi(float statevars[MAXSTATEVARS]){
	unsigned int idim,iidim;
	float elem,elemi,elem0,coord,*coords;
	NeuronModelTable *tab;
	NeuronModelTable::TableDimension *dim;
	float coeints[MAXSTATEVARS];
	void **dpointers[MAXSTATEVARS],**dpointer;
	int tableinds[MAXSTATEVARS];

	tab=this;
	dpointers[0]=(void **)tab->elems;

	for(idim=0;idim<tab->ndims;idim++){
		dim=&tab->dims[idim];
		coord=statevars[dim->statevar];
		if(dim->interp){
			coords=dim->coord;
			if(coord>last_coord(dim)){
				tableinds[idim]=dim->size-2;
				coeints[idim]=1;
			}else{
				if(coord<dim->vfirst){
					tableinds[idim]=0;
					coeints[idim]=0;
				}else{
					tableinds[idim]=table_ind_int(dim,coord);
					coeints[idim]=((coord-coords[tableinds[idim]])/(coords[tableinds[idim]+1]-coords[tableinds[idim]]));
				}
			}
		}else{
			tableinds[idim]=table_indcomp2(dim,coord);
		}
	}
	
	for(idim=0;idim<tab->ndims-1;idim++){
		dpointers[idim+1]=(void **)dpointers[idim][tableinds[idim]];
	}
	
	elemi=elem0=((float *)dpointers[idim])[tableinds[idim]];

	for(iidim=tab->firstintdim;iidim>=0;iidim-=tab->dims[iidim].nextintdim){
		dpointer=dpointers[iidim];
		for(idim=iidim;idim<tab->ndims-1;idim++){
			dpointer=(void **)dpointer[tableinds[idim]+(idim==iidim)];
		}
		elem=((float *)dpointer)[tableinds[idim]+(idim==iidim)];
		elemi+=(elem-elem0)*coeints[iidim];
	}
	
	return(elemi);
}

// Lineal interpolation-extrapolation
float NeuronModelTable::TableAccessInterpLiEx(float statevars[MAXSTATEVARS]){
	unsigned int idim,iidim;
	float elem,elemi,elem0,coord,*coords;
	NeuronModelTable *tab;
	NeuronModelTable::TableDimension *dim;
	float coeints[MAXSTATEVARS];
	void **dpointers[MAXSTATEVARS],**dpointer;
	int tableinds[MAXSTATEVARS];

	tab=this;
	dpointers[0]=(void **)tab->elems;

	for(idim=0;idim<tab->ndims;idim++){
		dim=&tab->dims[idim];
		coord=statevars[dim->statevar];
		if(dim->interp){
			coords=dim->coord;
			if(coord>last_coord(dim)){
				tableinds[idim]=dim->size-2;
			}else{
				if(coord<dim->vfirst){
					tableinds[idim]=0;
				}else{
					tableinds[idim]=table_ind_int(dim,coord);
				}
			}
			
			coeints[idim]=((coord-coords[tableinds[idim]])/(coords[tableinds[idim]+1]-coords[tableinds[idim]]));
		}else{
			tableinds[idim]=table_indcomp2(dim,coord);
		}
	}
	
	for(idim=0;idim<tab->ndims-1;idim++){
		dpointers[idim+1]=(void **)dpointers[idim][tableinds[idim]];
	}
   
	elemi=elem0=((float *)dpointers[idim])[tableinds[idim]];

	for(iidim=tab->firstintdim;iidim>=0;iidim-=tab->dims[iidim].nextintdim){
		dpointer=dpointers[iidim];
		for(idim=iidim;idim<tab->ndims-1;idim++){
			dpointer=(void **)dpointer[tableinds[idim]+(idim==iidim)];
		}
		elem=((float *)dpointer)[tableinds[idim]+(idim==iidim)];
		elemi+=(elem-elem0)*coeints[iidim];
	}
	
	return(elemi);
}

// 2-position lineal interpolation
float NeuronModelTable::TableAccessInterp2Li(float statevars[MAXSTATEVARS]){
	unsigned int idim,iidim,nintdims,zpos;
	float elem,elemi,elem0,avepos,coord,*coords;
	NeuronModelTable *tab;
	NeuronModelTable::TableDimension *dim;
	float coeints[MAXSTATEVARS];
	void **dpointers[MAXSTATEVARS],**dpointer;
	int tableinds[MAXSTATEVARS];

	tab=this;
	dpointers[0]=(void **)tab->elems;

	avepos=0;
	nintdims=0;
	for(idim=0;idim<tab->ndims;idim++){
		dim=&tab->dims[idim];
		coord=statevars[dim->statevar];
		if(dim->interp){
			coords=dim->coord;
			if(coord>last_coord(dim)){
				tableinds[idim]=dim->size-2;
				coeints[idim]=1;
			}else{
				if(coord<dim->vfirst){
					tableinds[idim]=0;
					coeints[idim]=0;
				}else{
					tableinds[idim]=table_ind_int(dim,coord);
					coeints[idim]=((coord-coords[tableinds[idim]])/(coords[tableinds[idim]+1]-coords[tableinds[idim]]));
				}
			}
			
			avepos+=coeints[idim];
			nintdims++;
		}else{
			tableinds[idim]=table_indcomp2(dim,coord);
		}
	}
    
    zpos=(avepos/nintdims)>0.5;
    if(zpos){
    	for(iidim=tab->firstintdim;iidim>=0;iidim-=tab->dims[iidim].nextintdim){
    		tableinds[iidim]++;
    		coeints[iidim]=coeints[iidim]-1;
    	}
    }
    
    zpos=zpos*-2+1; // pos=1 or -1
    for(idim=0;idim<tab->ndims-1;idim++){
    	dpointers[idim+1]=(void **)dpointers[idim][tableinds[idim]];
    }
    
    elemi=elem0=((float *)dpointers[idim])[tableinds[idim]];

	for(iidim=tab->firstintdim;iidim>=0;iidim-=tab->dims[iidim].nextintdim){
		dpointer=dpointers[iidim];
		for(idim=iidim;idim<tab->ndims-1;idim++){
			dpointer=(void **)dpointer[tableinds[idim]+zpos*(idim==iidim)];
		}
		
		elem=((float *)dpointer)[tableinds[idim]+zpos*(idim==iidim)];
		elemi+=(elem-elem0)*coeints[iidim];
	}
   
	return(elemi);
}

// n-position lineal interpolation
float NeuronModelTable::TableAccessInterpNLi(float statevars[MAXSTATEVARS]){
	int idim,iidim;
	float elem,elemi,elem0,coord,*coords;
	NeuronModelTable *tab;
	NeuronModelTable::TableDimension *dim;
	int intstate[MAXSTATEVARS];
	float coeints[MAXSTATEVARS];
	void **dpointers[MAXSTATEVARS],**dpointer;
	int tableinds[MAXSTATEVARS];

	tab=this;
	dpointers[0]=(void **)tab->elems;

	for(idim=0;idim<tab->ndims;idim++){
		dim=&tab->dims[idim];
		coord=statevars[dim->statevar];
		if(dim->interp){
			coords=dim->coord;
			if(coord>last_coord(dim)){
				tableinds[idim]=dim->size-2;
				coeints[idim]=1;
			}else{
				if(coord<dim->vfirst){
					tableinds[idim]=0;
					coeints[idim]=0;
				}else{
					tableinds[idim]=table_ind_int(dim,coord);
					coeints[idim]=((coord-coords[tableinds[idim]])/(coords[tableinds[idim]+1]-coords[tableinds[idim]]));
				}
			}
			
			if(coeints[idim]>0.5){
				coeints[idim]=1-coeints[idim];
				intstate[idim]=1;
			}else{
				intstate[idim]=0;
			}
		}else{
			tableinds[idim]=table_indcomp2(dim,coord);
			intstate[idim]=0;
        }
	}
   
	for(idim=0;idim<tab->ndims-1;idim++){
		dpointers[idim+1]=(void **)dpointers[idim][tableinds[idim]+intstate[idim]];
	}
	
	elemi=elem0=((float *)dpointers[idim])[tableinds[idim]+intstate[idim]];

	for(iidim=tab->firstintdim;iidim>=0;iidim-=tab->dims[iidim].nextintdim){
		dpointer=dpointers[iidim];
		for(idim=iidim;idim<tab->ndims-1;idim++){
			dpointer=(void **)dpointer[tableinds[idim]+(intstate[idim]^(idim==iidim))];
		}
		
		elem=((float *)dpointer)[tableinds[idim]+(intstate[idim]^(idim==iidim))];
		elemi+=(elem-elem0)*coeints[iidim];
	}
	
	return(elemi);
}

float NeuronModelTable::TableAccess(float *statevars){
	function funcArr1[] = {
		&NeuronModelTable::TableAccessDirect,
      	&NeuronModelTable::TableAccessInterpBi,
      	&NeuronModelTable::TableAccessInterpLi,
      	&NeuronModelTable::TableAccessInterpLiEx,
      	&NeuronModelTable::TableAccessInterp2Li,
      	&NeuronModelTable::TableAccessInterpNLi};
	/*float (NeuronType::*table_access_fn[])(int , Neuron *)={
      	&NeuronType::table_access_direct,
      	&NeuronType::table_access_interp_bi,
      	&NeuronType::table_access_interp_li,
      	&NeuronType::table_access_interp_li_ex,
      	&NeuronType::table_access_interp_2li,
      	&NeuronType::table_access_interp_nli};
   	return((table_access_fn[this->tables[ntab].interp])(ntab,neu));*/
   	return(this->*funcArr1[this->interp])(statevars);
}
