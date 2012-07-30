/***************************************************************************
 *                           LIFTimeDrivenModel.cpp                        *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
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

//#include "../../include/neuron_model/LIFTimeDrivenModel_GPU.h"
//#include "../../include/neuron_model/NeuronState.h"
//
//#include <iostream>
//#include <cmath>
//#include <string>
//
//#include "../../include/spike/EDLUTFileException.h"
//#include "../../include/spike/Neuron.h"
//#include "../../include/spike/InternalSpike.h"
//#include "../../include/spike/PropagatedSpike.h"
//#include "../../include/spike/Interconnection.h"
//
//#include "../../include/simulation/Utils.h"

		#include "../../include/neuron_model/LIFTimeDrivenModel_CUDA.h"
		#include "../../include/cudaError.h"
		//Library for CUDA
		#include <cutil_inline.h>



cudaEvent_t synchronize;

void createSynchronize(){
	HANDLE_ERROR(cudaEventCreate(&synchronize));
}

void synchronizeGPU_CPU(){
	HANDLE_ERROR(cudaEventRecord(synchronize,0));
	HANDLE_ERROR(cudaEventSynchronize(synchronize));
}

void destroySynchronize(){
	//HANDLE_ERROR(cudaEventDestroy(synchronize));
}


__global__ void UpdateState(float * parameter, float * AuxStateGPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, int SizeStates, double CurrentTime){
	float inv_param_4=1/parameter[4];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
	int index2, index3;

	double elapsed_time =CurrentTime - LastUpdateGPU[index];
	float elapsed_time1 =elapsed_time;

	float exponential1=exp(-(elapsed_time1/parameter[5]));
	float exponential2=exp(-(elapsed_time1/parameter[6]));

	while (index<SizeStates){
		index2 = index*2;
		index3 = index*3;

		double last_spike=LastSpikeTimeGPU[index]+elapsed_time;
		LastSpikeTimeGPU[index]+=elapsed_time;

		float vm = StateGPU[index3];
		float gexc = StateGPU[index3+1]+AuxStateGPU[index2];
		float ginh = StateGPU[index3+2]+AuxStateGPU[index2+1];

		bool spike=false;

		if (last_spike > parameter[7]) {
			vm = vm + elapsed_time * ( gexc * (parameter[0] - vm) + ginh * (parameter[1] - vm) + parameter[8] * (parameter[2] - vm))*inv_param_4;
			if (vm > parameter[3]){
				LastSpikeTimeGPU[index]=0;
				spike = true;
				vm = parameter[2];
			}
		}

		InternalSpikeGPU[index]=spike;
	
		gexc = gexc * exponential1;
		ginh = ginh * exponential2;

		StateGPU[index3]=vm;
		StateGPU[index3+1]=gexc;
		StateGPU[index3+2]=ginh;
        LastUpdateGPU[index]=CurrentTime;

		index+=blockDim.x*gridDim.x;
	}
}


//__global__ void UpdateState(float * parameter, float * AuxStateGPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, int SizeStates, double CurrentTime){
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//	int index2, index3;
//
//	while (index<SizeStates){
//		index2 = index*2;
//		index3 = index*3;
//
//		double elapsed_time =CurrentTime - LastUpdateGPU[index];
//		float elapsed_time1 =elapsed_time;
//		LastSpikeTimeGPU[index]+=elapsed_time;
//		double last_spike=LastSpikeTimeGPU[index];
//
//		float vm = StateGPU[index3];
//		float gexc = StateGPU[index3+1]+AuxStateGPU[index2];
//		float ginh = StateGPU[index3+2]+AuxStateGPU[index2+1];
//
//		bool spike=false;
//
//		if (last_spike > parameter[7]) {
//			vm = vm + elapsed_time * ( gexc * (parameter[0] - vm) + ginh * (parameter[1] - vm) + parameter[8] * (parameter[2] - vm))/parameter[4];
//			if (vm > parameter[3]){
//				LastSpikeTimeGPU[index]=0;
//				spike = true;
//				vm = parameter[2];
//			}
//		}
//
//		InternalSpikeGPU[index]=spike;
//	
//		gexc = gexc * exp(-(elapsed_time1/parameter[5]));
//		ginh = ginh * exp(-(elapsed_time1/parameter[6]));
//
//		StateGPU[index3]=vm;
//		StateGPU[index3+1]=gexc;
//		StateGPU[index3+2]=ginh;
//        LastUpdateGPU[index]=CurrentTime;
//
//		index+=blockDim.x*gridDim.x;
//	}
//}


void UpdateStateGPU(float * parameter, float * AuxStateGPU, float * AuxStateCPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, bool * InternalSpikeCPU, int SizeStates, double CurrentTime){
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties( &prop, 0 ));

	int N_thread, N_block;
	
    //GPU can use MapHostMemory
    if(prop.canMapHostMemory){
        N_thread = 128;
		N_block=prop.multiProcessorCount*4;
		if((SizeStates+N_thread-1)/N_thread < N_block){
			N_block = (SizeStates+N_thread-1)/N_thread;
		}
		UpdateState<<<N_block,N_thread>>>(parameter, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates, CurrentTime);

    }

    //GPU can transfer memory and execute kernel at same time.
	else if(prop.deviceOverlap){
		N_thread = 128;
		N_block=prop.multiProcessorCount*4;
		if((SizeStates+N_thread-1)/N_thread < N_block){
			N_block = (SizeStates+N_thread-1)/N_thread;
		}

		const int N_Stream=4;
		
		cudaStream_t stream[N_Stream];
		for (int i = 0; i < N_Stream; ++i){
			HANDLE_ERROR(cudaStreamCreate(&stream[i]));
		}

		int size[N_Stream];
		int offset[N_Stream];

		int N_Stream_use;
		int aux=SizeStates/(N_thread*N_block);
		if(aux<N_Stream){
			if(aux==0){
				N_Stream_use=1;
			}else{
				N_Stream_use=aux;
			}
			for (int i = 0; i < N_Stream_use; ++i){
				offset[i]=i*N_thread*N_block;
				if(i==(N_Stream_use-1)){
					size[i]=SizeStates-offset[i];
				}else{
					size[i]=N_thread*N_block;
				}
			}
		}else{
			N_Stream_use=N_Stream;
			for (int i = 0; i < N_Stream_use; ++i){
				offset[i]=i*N_thread*N_block * (aux/N_Stream_use);
				if(i==(N_Stream_use-1)){
					size[i]=SizeStates-offset[i];
				}else{
					size[i]=N_thread*N_block * (aux/N_Stream_use);
				}
			}
		}

		HANDLE_ERROR(cudaMemcpyAsync(AuxStateGPU, AuxStateCPU, sizeof(float)*2*size[0] , cudaMemcpyHostToDevice, stream[0]));
		for (int i = 0; i < N_Stream_use; ++i) {
			if((i+1)<N_Stream_use){
				HANDLE_ERROR(cudaMemcpyAsync(AuxStateGPU + offset[i+1] * 2, AuxStateCPU + offset[i+1] * 2, sizeof(float)*2*size[i+1] , cudaMemcpyHostToDevice, stream[i+1]));
			}
			UpdateState<<<N_block,N_thread,0,stream[i]>>>(parameter, AuxStateGPU+ offset[i] * 2, StateGPU+ offset[i] * 3, LastUpdateGPU + offset[i], LastSpikeTimeGPU + offset[i], InternalSpikeGPU + offset[i], size[i], CurrentTime);
			HANDLE_ERROR(cudaMemcpyAsync(InternalSpikeCPU + offset[i], InternalSpikeGPU + offset[i], sizeof(bool)*size[i],cudaMemcpyDeviceToHost, stream[i]));
		}
		for (int i = 0; i < N_Stream; ++i){
			HANDLE_ERROR(cudaStreamDestroy(stream[i]));
		}
	}
	
    //GPU uses memory transferences
	else{
		HANDLE_ERROR(cudaMemcpy(AuxStateGPU,AuxStateCPU,2*SizeStates*sizeof(float),cudaMemcpyHostToDevice));
		N_thread = 128;
		N_block=prop.multiProcessorCount*4;
		if((SizeStates+N_thread-1)/N_thread < N_block){
			N_block = (SizeStates+N_thread-1)/N_thread;
		}
		UpdateState<<<N_block,N_thread>>>(parameter, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates, CurrentTime);
		HANDLE_ERROR(cudaMemcpy(InternalSpikeCPU,InternalSpikeGPU,SizeStates*sizeof(bool),cudaMemcpyDeviceToHost));
	}

	HANDLE_ERROR(cudaDeviceSynchronize());
}


void UpdateStateGPU(float * elapsed_time, float * parameter, float * AuxStateGPU, float * AuxStateCPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, bool * InternalSpikeCPU, int SizeStates, double CurrentTime){
    cudaEvent_t start, end;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));

    HANDLE_ERROR(cudaEventRecord(start,0));	
	
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties( &prop, 0 ));

	int N_thread, N_block;

    //GPU can use MapHostMemory
    if(prop.canMapHostMemory){
        N_thread = 128;
        N_block=prop.multiProcessorCount*4;
        if((SizeStates+N_thread-1)/N_thread < N_block){
            N_block = (SizeStates+N_thread-1)/N_thread;
        }
        UpdateState<<<N_block,N_thread>>>(parameter, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates, CurrentTime);

    }

    //GPU can transfer memory and execute kernel at same time.
    else{ 
        if(prop.deviceOverlap){
		    N_thread = 128;
		    N_block=prop.multiProcessorCount*4;
		    if((SizeStates+N_thread-1)/N_thread < N_block){
			    N_block = (SizeStates+N_thread-1)/N_thread;
		    }

		    const int N_Stream=4;
    		
		    cudaStream_t stream[N_Stream];
		    for (int i = 0; i < N_Stream; ++i){
			    HANDLE_ERROR(cudaStreamCreate(&stream[i]));
		    }

		    int size[N_Stream];
		    int offset[N_Stream];

		    int N_Stream_use;
		    int aux=SizeStates/(N_thread*N_block);
		    if(aux<N_Stream){
			    if(aux==0){
				    N_Stream_use=1;
			    }else{
				    N_Stream_use=aux;
			    }
			    for (int i = 0; i < N_Stream_use; ++i){
				    offset[i]=i*N_thread*N_block;
				    if(i==(N_Stream_use-1)){
					    size[i]=SizeStates-offset[i];
				    }else{
					    size[i]=N_thread*N_block;
				    }
			    }
		    }else{
			    N_Stream_use=N_Stream;
			    for (int i = 0; i < N_Stream_use; ++i){
				    offset[i]=i*N_thread*N_block * (aux/N_Stream_use);
				    if(i==(N_Stream_use-1)){
					    size[i]=SizeStates-offset[i];
				    }else{
					    size[i]=N_thread*N_block * (aux/N_Stream_use);
				    }
			    }
		    }

		    HANDLE_ERROR(cudaMemcpyAsync(AuxStateGPU, AuxStateCPU, sizeof(float)*2*size[0] , cudaMemcpyHostToDevice, stream[0]));
		    for (int i = 0; i < N_Stream_use; ++i) {
			    if((i+1)<N_Stream_use){
				    HANDLE_ERROR(cudaMemcpyAsync(AuxStateGPU + offset[i+1] * 2, AuxStateCPU + offset[i+1] * 2, sizeof(float)*2*size[i+1] , cudaMemcpyHostToDevice, stream[i+1]));
			    }
			    UpdateState<<<N_block,N_thread,0,stream[i]>>>(parameter, AuxStateGPU+ offset[i] * 2, StateGPU+ offset[i] * 3, LastUpdateGPU + offset[i], LastSpikeTimeGPU + offset[i], InternalSpikeGPU + offset[i], size[i], CurrentTime);
			    HANDLE_ERROR(cudaMemcpyAsync(InternalSpikeCPU + offset[i], InternalSpikeGPU + offset[i], sizeof(bool)*size[i],cudaMemcpyDeviceToHost, stream[i]));
		    }
		    for (int i = 0; i < N_Stream; ++i){
			    HANDLE_ERROR(cudaStreamDestroy(stream[i]));
		    }
	    }
    	
        //GPU uses memory transferences
	    else{
		    HANDLE_ERROR(cudaMemcpy(AuxStateGPU,AuxStateCPU,2*SizeStates*sizeof(float),cudaMemcpyHostToDevice));
		    N_thread = 128;
		    N_block=prop.multiProcessorCount*4;
		    if((SizeStates+N_thread-1)/N_thread < N_block){
			    N_block = (SizeStates+N_thread-1)/N_thread;
		    }
		    UpdateState<<<N_block,N_thread>>>(parameter, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates, CurrentTime);
		    HANDLE_ERROR(cudaMemcpy(InternalSpikeCPU,InternalSpikeGPU,SizeStates*sizeof(bool),cudaMemcpyDeviceToHost));
	    }
    }

	HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaEventRecord(end,0));
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(elapsed_time,start,end));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));
    //printf("Elapsed time: %f\n",*elapsed_time);
}


__global__ void UpdateStateRK(float * parameter, float * AuxStateGPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, int SizeStates, double CurrentTime){
	float inv_param_4=1/parameter[4];
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int index2, index3;
	double elapsed_time =CurrentTime - LastUpdateGPU[index];
	
	float elapsed_time1 =elapsed_time;
	float elapsed_time2 =elapsed_time/2;

	
	float exponential1=exp(-(elapsed_time1/parameter[5]));
	float exponential2=exp(-(elapsed_time1/parameter[6]));

	float exponential3=exp(-((elapsed_time2)/parameter[5]));
	float exponential4=exp(-((elapsed_time2)/parameter[6]));

	while (index<SizeStates){
		index2 = index*2;
		index3 = index*3;
	
		double last_spike=LastSpikeTimeGPU[index]+elapsed_time;
		LastSpikeTimeGPU[index]+=elapsed_time;

		float vm = StateGPU[index3];
		float gexc = StateGPU[index3+1]+AuxStateGPU[index2];
		float ginh = StateGPU[index3+2]+AuxStateGPU[index2+1];

		bool spike=false;


		float nextgexc = gexc * exponential1;
		float nextginh = ginh * exponential2;
		if (last_spike > parameter[7]) {
			// 4th order Runge-Kutta terms
			// 1st term
			float k1 = (gexc * (parameter[0] - vm) + ginh * (parameter[1] - vm) + parameter[8] * (parameter[2]-vm))*inv_param_4;

			// 2nd term
			float gexcaux = gexc * exponential3;
			float ginhaux = ginh * exponential4;
			float yaux = vm+(k1*elapsed_time/2);
			float k2 = (gexcaux * (parameter[0] - yaux) + ginhaux * (parameter[1] - yaux) + parameter[8] * (parameter[2] - yaux))*inv_param_4;

			// 3rd term
			yaux = vm+(k2*elapsed_time/2);
			float k3 = (gexcaux * (parameter[0] - yaux) + ginhaux * (parameter[1] - yaux) + parameter[8] * (parameter[2] - yaux))*inv_param_4;

			// 4rd term
			gexcaux = nextgexc;
			ginhaux = nextginh;
			yaux = vm+(k3*elapsed_time);
			float k4 = (gexcaux * (parameter[0] - yaux) + ginhaux * (parameter[1] - yaux) + parameter[8] * (parameter[2] - yaux))*inv_param_4;

			vm = vm + (k1+2*(k2+k3)+k4)*elapsed_time/6;

			if (vm > parameter[3]){
				LastSpikeTimeGPU[index]=0;
				spike = true;
				vm = parameter[2];
			}
		}

		InternalSpikeGPU[index]=spike;
	
		StateGPU[index3]=vm;
		StateGPU[index3+1]=nextgexc;
		StateGPU[index3+2]=nextginh;
        LastUpdateGPU[index]=CurrentTime;

		index+=blockDim.x*gridDim.x;
	}
}

//__global__ void UpdateStateRK(float * parameter, float * AuxStateGPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, int SizeStates, double CurrentTime){
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//	int index2, index3;
//
//	while (index<SizeStates){
//		index2 = index*2;
//		index3 = index*3;
//
//		double elapsed_time =CurrentTime - LastUpdateGPU[index];
//		LastSpikeTimeGPU[index]+=elapsed_time;
//		double last_spike=LastSpikeTimeGPU[index];
//
//		float vm = StateGPU[index3];
//		float gexc = StateGPU[index3+1]+AuxStateGPU[index2];
//		float ginh = StateGPU[index3+2]+AuxStateGPU[index2+1];
//
//		bool spike=false;
//
//
//		float nextgexc = gexc * exp(-(elapsed_time/parameter[5]));
//		float nextginh = ginh * exp(-(elapsed_time/parameter[6]));
//
//		if (last_spike > parameter[7]) {
//			// 4th order Runge-Kutta terms
//			// 1st term
//			float k1 = (gexc * (parameter[0] - vm) + ginh * (parameter[1] - vm) + parameter[8] * (parameter[2]-vm))/parameter[4];
//
//			// 2nd term
//			float gexcaux = gexc * exp(-((elapsed_time/2)/parameter[5]));
//			float ginhaux = ginh * exp(-((elapsed_time/2)/parameter[6]));
//			float yaux = vm+(k1*elapsed_time/2);
//			float k2 = (gexcaux * (parameter[0] - yaux) + ginhaux * (parameter[1] - yaux) + parameter[8] * (parameter[2] - yaux))/parameter[4];
//
//			// 3rd term
//			yaux = vm+(k2*elapsed_time/2);
//			float k3 = (gexcaux * (parameter[0] - yaux) + ginhaux * (parameter[1] - yaux) + parameter[8] * (parameter[2] - yaux))/parameter[4];
//
//			// 4rd term
//			gexcaux = nextgexc;
//			ginhaux = nextginh;
//			yaux = vm+(k3*elapsed_time);
//			float k4 = (gexcaux * (parameter[0] - yaux) + ginhaux * (parameter[1] - yaux) + parameter[8] * (parameter[2] - yaux))/parameter[4];
//
//			vm = vm + (k1+2*k2+2*k3+k4)*elapsed_time/6;
//
//			if (vm > parameter[3]){
//				LastSpikeTimeGPU[index]=0;
//				spike = true;
//				vm = parameter[2];
//			}
//		}
//
//		InternalSpikeGPU[index]=spike;
//	
//		StateGPU[index3]=vm;
//		StateGPU[index3+1]=nextgexc;
//		StateGPU[index3+2]=nextginh;
//        LastUpdateGPU[index]=CurrentTime;
//
//		index+=blockDim.x*gridDim.x;
//	}
//}


void UpdateStateRKGPU(float * parameter, float * AuxStateGPU, float * AuxStateCPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, bool * InternalSpikeCPU, int SizeStates, double CurrentTime){
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties( &prop, 0 ));

	int N_thread, N_block;
	
    //GPU can use MapHostMemory
    if(prop.canMapHostMemory){
        N_thread = 128;
        N_block=prop.multiProcessorCount*4;
        if((SizeStates+N_thread-1)/N_thread < N_block){
            N_block = (SizeStates+N_thread-1)/N_thread;
        }
        UpdateStateRK<<<N_block,N_thread>>>(parameter, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates, CurrentTime);

    }

    //GPU can transfer memory and execute kernel at same time.
    else{ 
        if(prop.deviceOverlap){
		    N_thread = 128;
		    N_block=prop.multiProcessorCount*4;
		    if((SizeStates+N_thread-1)/N_thread < N_block){
			    N_block = (SizeStates+N_thread-1)/N_thread;
		    }

		    const int N_Stream=4;
    		
		    cudaStream_t stream[N_Stream];
		    for (int i = 0; i < N_Stream; ++i){
			    HANDLE_ERROR(cudaStreamCreate(&stream[i]));
		    }

		    int size[N_Stream];
		    int offset[N_Stream];

		    int N_Stream_use;
		    int aux=SizeStates/(N_thread*N_block);
		    if(aux<N_Stream){
			    if(aux==0){
				    N_Stream_use=1;
			    }else{
				    N_Stream_use=aux;
			    }
			    for (int i = 0; i < N_Stream_use; ++i){
				    offset[i]=i*N_thread*N_block;
				    if(i==(N_Stream_use-1)){
					    size[i]=SizeStates-offset[i];
				    }else{
					    size[i]=N_thread*N_block;
				    }
			    }
		    }else{
			    N_Stream_use=N_Stream;
			    for (int i = 0; i < N_Stream_use; ++i){
				    offset[i]=i*N_thread*N_block * (aux/N_Stream_use);
				    if(i==(N_Stream_use-1)){
					    size[i]=SizeStates-offset[i];
				    }else{
					    size[i]=N_thread*N_block * (aux/N_Stream_use);
				    }
			    }
		    }

		    HANDLE_ERROR(cudaMemcpyAsync(AuxStateGPU, AuxStateCPU, sizeof(float)*2*size[0] , cudaMemcpyHostToDevice, stream[0]));
		    for (int i = 0; i < N_Stream_use; ++i) {
			    if((i+1)<N_Stream_use){
				    HANDLE_ERROR(cudaMemcpyAsync(AuxStateGPU + offset[i+1] * 2, AuxStateCPU + offset[i+1] * 2, sizeof(float)*2*size[i+1] , cudaMemcpyHostToDevice, stream[i+1]));
			    }
			    UpdateStateRK<<<N_block,N_thread,0,stream[i]>>>(parameter, AuxStateGPU+ offset[i] * 2, StateGPU+ offset[i] * 3, LastUpdateGPU + offset[i], LastSpikeTimeGPU + offset[i], InternalSpikeGPU + offset[i], size[i], CurrentTime);
			    HANDLE_ERROR(cudaMemcpyAsync(InternalSpikeCPU + offset[i], InternalSpikeGPU + offset[i], sizeof(bool)*size[i],cudaMemcpyDeviceToHost, stream[i]));
		    }
		    for (int i = 0; i < N_Stream; ++i){
			    HANDLE_ERROR(cudaStreamDestroy(stream[i]));
		    }
	    }
    	
        //GPU uses memory transferences
	    else{
		    HANDLE_ERROR(cudaMemcpy(AuxStateGPU,AuxStateCPU,2*SizeStates*sizeof(float),cudaMemcpyHostToDevice));
		    N_thread = 128;
		    N_block=prop.multiProcessorCount*4;
		    if((SizeStates+N_thread-1)/N_thread < N_block){
			    N_block = (SizeStates+N_thread-1)/N_thread;
		    }
		    UpdateStateRK<<<N_block,N_thread>>>(parameter, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates, CurrentTime);
		    HANDLE_ERROR(cudaMemcpy(InternalSpikeCPU,InternalSpikeGPU,SizeStates*sizeof(bool),cudaMemcpyDeviceToHost));
	    }
    }
	HANDLE_ERROR(cudaDeviceSynchronize());
}




void UpdateStateRKGPU(float * elapsed_time, float * parameter, float * AuxStateGPU, float * AuxStateCPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, bool * InternalSpikeCPU, int SizeStates, double CurrentTime){
    cudaEvent_t start, end;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));

    HANDLE_ERROR(cudaEventRecord(start,0));	
	
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties( &prop, 0 ));

	int N_thread, N_block;

    //GPU can use MapHostMemory
    if(prop.canMapHostMemory){
        N_thread = 128;
        N_block=prop.multiProcessorCount*4;
        if((SizeStates+N_thread-1)/N_thread < N_block){
            N_block = (SizeStates+N_thread-1)/N_thread;
        }
        UpdateStateRK<<<N_block,N_thread>>>(parameter, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates, CurrentTime);

    }

    //GPU can transfer memory and execute kernel at same time.
    else{ 
        if(prop.deviceOverlap){
		    N_thread = 128;
		    N_block=prop.multiProcessorCount*4;
		    if((SizeStates+N_thread-1)/N_thread < N_block){
			    N_block = (SizeStates+N_thread-1)/N_thread;
		    }

		    const int N_Stream=4;
    		
		    cudaStream_t stream[N_Stream];
		    for (int i = 0; i < N_Stream; ++i){
			    HANDLE_ERROR(cudaStreamCreate(&stream[i]));
		    }

		    int size[N_Stream];
		    int offset[N_Stream];

		    int N_Stream_use;
		    int aux=SizeStates/(N_thread*N_block);
		    if(aux<N_Stream){
			    if(aux==0){
				    N_Stream_use=1;
			    }else{
				    N_Stream_use=aux;
			    }
			    for (int i = 0; i < N_Stream_use; ++i){
				    offset[i]=i*N_thread*N_block;
				    if(i==(N_Stream_use-1)){
					    size[i]=SizeStates-offset[i];
				    }else{
					    size[i]=N_thread*N_block;
				    }
			    }
		    }else{
			    N_Stream_use=N_Stream;
			    for (int i = 0; i < N_Stream_use; ++i){
				    offset[i]=i*N_thread*N_block * (aux/N_Stream_use);
				    if(i==(N_Stream_use-1)){
					    size[i]=SizeStates-offset[i];
				    }else{
					    size[i]=N_thread*N_block * (aux/N_Stream_use);
				    }
			    }
		    }

		    HANDLE_ERROR(cudaMemcpyAsync(AuxStateGPU, AuxStateCPU, sizeof(float)*2*size[0] , cudaMemcpyHostToDevice, stream[0]));
		    for (int i = 0; i < N_Stream_use; ++i) {
			    if((i+1)<N_Stream_use){
				    HANDLE_ERROR(cudaMemcpyAsync(AuxStateGPU + offset[i+1] * 2, AuxStateCPU + offset[i+1] * 2, sizeof(float)*2*size[i+1] , cudaMemcpyHostToDevice, stream[i+1]));
			    }
			    UpdateStateRK<<<N_block,N_thread,0,stream[i]>>>(parameter, AuxStateGPU+ offset[i] * 2, StateGPU+ offset[i] * 3, LastUpdateGPU + offset[i], LastSpikeTimeGPU + offset[i], InternalSpikeGPU + offset[i], size[i], CurrentTime);
			    HANDLE_ERROR(cudaMemcpyAsync(InternalSpikeCPU + offset[i], InternalSpikeGPU + offset[i], sizeof(bool)*size[i],cudaMemcpyDeviceToHost, stream[i]));
		    }
		    for (int i = 0; i < N_Stream; ++i){
			    HANDLE_ERROR(cudaStreamDestroy(stream[i]));
		    }
	    }
    	
        //GPU uses memory transferences
	    else{
		    HANDLE_ERROR(cudaMemcpy(AuxStateGPU,AuxStateCPU,2*SizeStates*sizeof(float),cudaMemcpyHostToDevice));
		    N_thread = 128;
		    N_block=prop.multiProcessorCount*4;
		    if((SizeStates+N_thread-1)/N_thread < N_block){
			    N_block = (SizeStates+N_thread-1)/N_thread;
		    }
		    UpdateStateRK<<<N_block,N_thread>>>(parameter, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates, CurrentTime);
		    HANDLE_ERROR(cudaMemcpy(InternalSpikeCPU,InternalSpikeGPU,SizeStates*sizeof(bool),cudaMemcpyDeviceToHost));
	    }
    }

	HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaEventRecord(end,0));
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(elapsed_time,start,end));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));
//printf("Elapsed time: %f\n",*elapsed_time);
}


void InformationGPU(){
	cudaDeviceProp prop;
	int count;
	HANDLE_ERROR(cudaGetDeviceCount( &count ));
	for (int i=0; i< count; i++) {
		cudaGetDeviceProperties( &prop, i ) ;
		printf(" --- General Information for device %d ---\n", i );
		printf ( "Name: %s\n ", prop.name );
		printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
		printf ( "Clock rate: %d\n", prop. clockRate );
		printf ( "Device copy overlap: ");
		if (prop.deviceOverlap)
			printf ( "Enabled\n" );
		else
			printf ( "Disabled\n" ) ;
		printf ( "Concurrent Kernels: ");
		if (prop.concurrentKernels)
			printf ( "Enabled\n" );
		else
			printf ( "Disabled\n" ) ;

		
		
		printf ( "Kernel execition timeout ");
		if (prop.kernelExecTimeoutEnabled)
			printf ( "Enabled\n" );
		else
			printf ( "Disabled\n" );
		
		printf("--- Memory Information for device %d ---\n", i );
		printf("Total global mem; %ld\n", prop. totalGlobalMem );
		printf("Total constant Mem: %ld\n", prop. totalConstMem );
		printf("Max mem pitch: %ld\n", prop.memPitch );
		printf("Texture Alignment: %ld\n", prop. textureAlignment );
		printf(" --- MP Information for device %d ---\n", i );
		printf ( "Multiprocessor count: %d\n",prop.multiProcessorCount );

		printf ( "Shared mem per mp: %ld\n", prop. sharedMemPerBlock );
		printf("Registers per rnp: %d\n", prop.regsPerBlock );
		printf("Threads in warp: %d\n", prop.warpSize ) ;
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

		printf ( "Max thread dimensions: (%d, %d, %d) \n",
			prop. maxThreadsDim[0], prop.maxThreadsDim[1] ,
			prop.maxThreadsDim[2] );
		printf( "Max grid dimensions: (%d, %d, %d\n",
			prop.maxGridSize[0], prop.maxGridSize[1] ,
			prop.maxGridSize[2] );
		printf ( "\n" );
	}

}