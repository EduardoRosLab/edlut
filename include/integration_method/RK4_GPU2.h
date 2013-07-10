/***************************************************************************
 *                           RK4_GPU2.h                                   *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Francisco Naveros                    *
 * email                : fnaveros@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef RK4_GPU2_H_
#define RK4_GPU2_H_

/*!
 * \file RK4_GPU2.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement a fourth order Runge-Kutta integration method in GPU (this class is stored
 * in GPU memory and executed in GPU. All integration methods in GPU are fixed step due to the parallel
 * architecture of this one.
 */

#include "./IntegrationMethod_GPU2.h"

#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU2.h"

//Library for CUDA
#include <helper_cuda.h>

/*!
 * \class RK4_GPU2
 *
 * \brief Fourth order Runge-Kutta integration method in GPU.
 * 
 * This class abstracts the behavior of a Euler integration method for neurons in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date May 2012
 */
class RK4_GPU2 : public IntegrationMethod_GPU2 {
	public:

		/*!
		 * \brief These vectors are used as auxiliar vectors.
		*/
		float * AuxNeuronState;
		float * AuxNeuronState1;
		float * AuxNeuronState2;
		float * AuxNeuronState3;
		float * AuxNeuronState4;


		/*!
		 * \brief Constructor of the class with 5 parameter.
		 *
		 * It generates a new fourth order Runge-Kutta object in GPU memory.
		 *
		 * \param N_neuronStateVariables Number of state variables for each cell.
		 * \param N_differentialNeuronState Number of state variables witch are calculate with a differential equation for each cell.
		 * \param N_timeDependentNeuronState Number of state variables witch ara calculate with a time dependent equation for each cell.
		 * \param Total_N_thread Number of thread in GPU (in this method it is not necessary)
		 * \param Buffer_GPU This vector contains all the necesary GPU memory witch have been reserved in the CPU (this memory
		 * could be reserved directly in the GPU, but this suppose some restriction in the amount of memory witch can be reserved).
		 */
		__device__ RK4_GPU2(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int Total_N_thread, void ** Buffer_GPU):IntegrationMethod_GPU2(N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, Total_N_thread){
			AuxNeuronState=((float*)Buffer_GPU[0]);
			AuxNeuronState1=((float*)Buffer_GPU[1]);
			AuxNeuronState2=((float*)Buffer_GPU[2]);
			AuxNeuronState3=((float*)Buffer_GPU[3]);
			AuxNeuronState4=((float*)Buffer_GPU[4]);
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ ~RK4_GPU2(){
		}
		


		/*!
		 * \brief It calculate the next neural state varaibles of the model.
		 *
		 * It calculate the next neural state varaibles of the model.
		 *
		 * \param index Index of the cell inside the neuron model for method with memory (e.g. BDF).
		 * \param SizeStates Number of neurons
		 * \param Model The NeuronModel.
		 * \param NeuronState Vector of neuron state variables for all neurons.
		 * \param elapsed_time integration time step.
		 */
		__device__ void NextDifferentialEcuationValue(int index, int SizeStates, TimeDrivenNeuronModel_GPU2 * Model, float * NeuronState, double elapsed_time){

			int offset1=gridDim.x * blockDim.x;
			int offset2=blockDim.x*blockIdx.x + threadIdx.x;

			//1st term
			Model->EvaluateDifferentialEcuation(index, SizeStates, NeuronState, AuxNeuronState1);
			
			//2nd term
			for (int j=0; j<N_DifferentialNeuronState; j++){
				AuxNeuronState[j*offset1 + offset2]= NeuronState[j*SizeStates + index] + AuxNeuronState1[j*offset1 + offset2]*elapsed_time*0.5f;
			}
			for (int j=N_DifferentialNeuronState; j<N_NeuronStateVariables; j++){
				AuxNeuronState[j*offset1 + offset2]=NeuronState[j*SizeStates + index];
			}


			Model->EvaluateTimeDependentEcuation(offset2, offset1, AuxNeuronState, elapsed_time*0.5f);
			Model->EvaluateDifferentialEcuation(offset2, offset1, AuxNeuronState, AuxNeuronState2);

			//3rd term
			for (int j=0; j<N_DifferentialNeuronState; j++){
				AuxNeuronState[j*offset1 + offset2]= NeuronState[j*SizeStates + index] + AuxNeuronState2[j*offset1 + offset2]*elapsed_time*0.5f;
			}

			Model->EvaluateDifferentialEcuation(offset2, offset1, AuxNeuronState, AuxNeuronState3);

			//4rd term
			for (int j=0; j<N_DifferentialNeuronState; j++){
				AuxNeuronState[j*offset1 + offset2]= NeuronState[j*SizeStates + index] + AuxNeuronState3[j*offset1 + offset2]*elapsed_time;
			}

			Model->EvaluateTimeDependentEcuation(offset2, offset1, AuxNeuronState, elapsed_time*0.5);
			Model->EvaluateDifferentialEcuation(offset2, offset1, AuxNeuronState, AuxNeuronState4);


			for (int j=0; j<N_DifferentialNeuronState; j++){
				NeuronState[j*SizeStates + index]+=(AuxNeuronState1[j*offset1 + offset2]+2*(AuxNeuronState2[j*offset1 + offset2]+AuxNeuronState3[j*offset1 + offset2])+AuxNeuronState4[j*offset1 + offset2])*elapsed_time*0.166666666667f;
			}

			//Finaly, we evaluate the neural state variables with time dependence.
			Model->EvaluateTimeDependentEcuation(index, SizeStates, NeuronState, elapsed_time);
		}




		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param index indicate witch neuron must be reseted.
		 *
		 */
		__device__ void resetState(int index){
		}

};



#endif /* RK4_GPU2_H_ */
