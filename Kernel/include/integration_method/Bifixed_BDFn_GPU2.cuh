/***************************************************************************
 *                           Bifixed_BDFn_GPU2.cuh                         *
 *                           -------------------                           *
 * copyright            : (C) 2020 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef BIFIXED_BDF_GPU2_H_
#define BIFIXED_BDF_GPU2_H_

/*!
 * \file Bifixed_BDFn_GPU2.cuh
 *
 * \author Francisco Naveros
 * \date April 2020
 *
 * This file declares a class which implements two multi step  BDF (Backward Differentiation Formulas) integration methods (
 * first and second order BDF integration method) in GPU. This method implements a progressive implementation of the
 * higher order integration method using the lower order integration mehtod (BDF1->BDF2).
 */

#include "integration_method/BifixedStep_GPU2.cuh"


//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class Bifixed_BDFn_GPU2
 *
 * \brief Bifixed_BDF1 and Bifixed_BDF2 integration methods in GPU.
 *
 * This class abstracts the behavior of BDF integration method for neurons in a
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date April 2020
 */
template <class Neuron_Model_GPU2>
class Bifixed_BDFn_GPU2 : public BifixedStep_GPU2<Neuron_Model_GPU2> {
	public:

		/*!
		 * \brief These vectors are used as auxiliar vectors.
		*/
		float * AuxNeuronState;
		float * AuxNeuronState_p;
		float * AuxNeuronState_p1;
		float * AuxNeuronState_c;
		float * jacnum;
		float * J;
		float * inv_J;
		//For Jacobian
		float * AuxNeuronState2;
		float * AuxNeuronState_pos;
		float * AuxNeuronState_neg;

		/*!
		 * \brief This constant matrix contains the coefficients of each order for the BDF integration mehtod.
		*/
		float * Coeficient;
			//{1.0f, 0.0f, 1.0f, 0.0f,
			//1.0f, 0.0f, 1.0f, 0.0f,
			//2.0f / 3.0f, 0.0f, 4.0f / 3.0f, -1.0f / 3.0f}


		/*!
		 * \brief This vector stores previous neuron state variable for all neuron. This one is used as a memory.
		*/
		float * PreviousNeuronState;

		/*!
		 * \brief This vector stores previous neuron state variable for all neuron. This one is used as a memory.
		*/
		float * D;

		/*!
		 * \brief This vector contains the state of each neuron (BDF order).
		*/
		int * state;

		/*!
		 * \brief This value stores the order of the integration method.
		*/
		int BDForder;


		/*!
		 * \brief Constructor of the class with 3 parameter.
		 *
		 * It generates a new BDF object in GPU memory indicating the order of the method.
		*
		* \param TimeDrivenNeuronModel pointer to the time driven neuron model
		 * \param Buffer_GPU This vector contains all the necesary GPU memory which have been reserved in the CPU (this memory
		 * could be reserved directly in the GPU, but this suppose some restriction in the amount of memory which can be reserved).
		 * \param BDForder BDF order (1 or 2).
		 */
		__device__ Bifixed_BDFn_GPU2(Neuron_Model_GPU2* NewModel, void ** Buffer_GPU):BifixedStep_GPU2<Neuron_Model_GPU2>(NewModel, Buffer_GPU){
			float * aux_integration_method_parameters_GPU = ((float*)Buffer_GPU[0]);
			BDForder = int(aux_integration_method_parameters_GPU[1]);

			AuxNeuronState = ((float*)Buffer_GPU[1]);
			AuxNeuronState_p = ((float*)Buffer_GPU[2]);
			AuxNeuronState_p1 = ((float*)Buffer_GPU[3]);
			AuxNeuronState_c = ((float*)Buffer_GPU[4]);
			jacnum = ((float*)Buffer_GPU[5]);
			J = ((float*)Buffer_GPU[6]);
			inv_J = ((float*)Buffer_GPU[7]);

			Coeficient=((float *)Buffer_GPU[8]);

			PreviousNeuronState = ((float *)Buffer_GPU[9]);
			D = ((float *)Buffer_GPU[10]);
			state = ((int *)Buffer_GPU[11]);

			AuxNeuronState2 = ((float*)Buffer_GPU[12]);
			AuxNeuronState_pos = ((float*)Buffer_GPU[13]);
			AuxNeuronState_neg = ((float*)Buffer_GPU[14]);
		}


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ ~Bifixed_BDFn_GPU2(){
		}


		/*!
		 * \brief It calculate the next neural state varaibles of the model.
		 *
		 * It calculate the next neural state varaibles of the model.
		 *
		 * \param index Index of the cell inside the neuron model for method with memory (e.g. BDF).
		 * \param SizeStates Number of neurons
		 * \param NeuronState Vector of neuron state variables for all neurons.
		 */
		__device__ void NextDifferentialEquationValues(int index, int SizeStates, float * NeuronState){
			int offset1 = gridDim.x * blockDim.x;
			int offset2 = blockDim.x*blockIdx.x + threadIdx.x;

			for (int iteration = 0; iteration < this->N_BifixedSteps; iteration++){
				float previous_V = NeuronState[index];
				
				//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
				if (state[index] == 0){
					this->neuron_model->EvaluateDifferentialEquation(index, SizeStates, NeuronState, AuxNeuronState, this->BifixedElapsedTimeInNeuronModelScale);
					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_p[j*offset1 + offset2] = NeuronState[j*SizeStates + index] + this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j*offset1 + offset2];
					}
				}
				//In this case we use the value of previous states to calculate an aproximation of the solution.
				else{
					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_p[j*offset1 + offset2] = NeuronState[j*SizeStates + index];
						for (int i = 0; i < state[index]; i++){
							AuxNeuronState_p[j*offset1 + offset2] += D[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
						}
					}
				}

				for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
					AuxNeuronState_p[i*offset1 + offset2] = NeuronState[i*SizeStates + index];
				}

				this->neuron_model->EvaluateTimeDependentEquation(offset2, offset1, AuxNeuronState_p, index, 0);

				float epsi = 1.0f;
				int k = 0;

				//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
				//epsi is the difference between two consecutive aproximation of the implicit method.
				while (epsi > 1e-16 && k < 5){
					this->neuron_model->EvaluateDifferentialEquation(offset2, offset1, AuxNeuronState_p, AuxNeuronState, this->BifixedElapsedTimeInNeuronModelScale);
					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_c[j*offset1 + offset2] = Coeficient[state[index] * 4 + 0] * this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j*offset1 + offset2] + Coeficient[state[index] * 4 + 1] * NeuronState[j*SizeStates + index];
						for (int i = 1; i < state[index]; i++){
							AuxNeuronState_c[j*offset1 + offset2] += Coeficient[state[index] * 4 + i + 1] * PreviousNeuronState[(i - 1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
						}
					}

					//jacobian.
					Jacobian(offset2, offset1, AuxNeuronState_p, jacnum);

					for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
						for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
							J[z*offset1*this->neuron_model->N_DifferentialNeuronState + t*offset1 + offset2] = Coeficient[state[index] * 4 + 0] * this->BifixedElapsedTimeInNeuronModelScale * jacnum[z*offset1*this->neuron_model->N_DifferentialNeuronState + t*offset1 + offset2];
							if (z == t){
								J[z*offset1*this->neuron_model->N_DifferentialNeuronState + t*offset1 + offset2] -= 1;
							}
						}
					}
					this->invermat(offset2, offset1, J, inv_J);

					for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
						float aux = 0.0;
						for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
							aux += inv_J[z*offset1*this->neuron_model->N_DifferentialNeuronState + t*offset1 + offset2] * (AuxNeuronState_p[t*offset1 + offset2] - AuxNeuronState_c[t*offset1 + offset2]);
						}
						AuxNeuronState_p1[z*offset1 + offset2] = aux + AuxNeuronState_p[z*offset1 + offset2];
					}

					//We calculate the difference between both aproximations.
					float aux = 0.0;
					float aux2 = 0.0;
					for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
						aux = fabs(AuxNeuronState_p1[z*offset1 + offset2] - AuxNeuronState_p[z*offset1 + offset2]);
						AuxNeuronState_p[z*offset1 + offset2] = AuxNeuronState_p1[z*offset1 + offset2];
						if (aux > aux2){
							aux2 = aux;
						}
					}
					epsi = aux2;
					k++;

				}

				//We increase the state of the integration method.
				if (state[index] < BDForder){
					state[index]++;
				}

				//We acumulate these new values for the next step.
				for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
					for (int i = (state[index] - 1); i > 0; i--){
						D[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = -D[(i - 1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
					}
					D[0 * SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = AuxNeuronState_p[j*offset1 + offset2] - NeuronState[j*SizeStates + index];
					for (int i = 1; i < state[index]; i++){
						D[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] += D[(i - 1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
					}
				}

				if (state[index]>1){
					for (int i = state[index] - 2; i > 0; i--){
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							PreviousNeuronState[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = PreviousNeuronState[(i - 1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
						}
					}
					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						PreviousNeuronState[0 * SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = NeuronState[j*SizeStates + index];
					}
				}
				for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
					NeuronState[z*SizeStates + index] = AuxNeuronState_p[z*offset1 + offset2];
				}

				//Finaly, we evaluate the neural state variables with time dependence.
				this->neuron_model->EvaluateTimeDependentEquation(index, SizeStates, NeuronState, index, 0);

				//Update the last spike time.
				this->neuron_model->vectorNeuronState_GPU2->LastSpikeTimeGPU[index] += this->BifixedElapsedTimeInSeconds;

				this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);
			}
		}

		/*!
		* \brief It calculate the next neural state variables of the model.
		*
		* It calculate the next neural state varaibles of the model.
		*
		* \param SizeStates Number of neurons
		* \param NeuronState Vector of neuron state variables for all neurons.
		*/
		__device__ virtual void NextDifferentialEquationValues(int SizeStates, float * NeuronState) {
			int offset1 = gridDim.x * blockDim.x;
			int offset2 = blockDim.x*blockIdx.x + threadIdx.x;
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			while (index < SizeStates){
				for (int iteration = 0; iteration < this->N_BifixedSteps; iteration++){
					float previous_V = NeuronState[index];

					//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
					if (state[index] == 0){
						this->neuron_model->EvaluateDifferentialEquation(index, SizeStates, NeuronState, AuxNeuronState, this->BifixedElapsedTimeInNeuronModelScale);
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_p[j*offset1 + offset2] = NeuronState[j*SizeStates + index] + this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j*offset1 + offset2];
						}
					}
					//In this case we use the value of previous states to calculate an aproximation of the solution.
					else{
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_p[j*offset1 + offset2] = NeuronState[j*SizeStates + index];
							for (int i = 0; i < state[index]; i++){
								AuxNeuronState_p[j*offset1 + offset2] += D[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
							}
						}
					}

					for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
						AuxNeuronState_p[i*offset1 + offset2] = NeuronState[i*SizeStates + index];
					}

					this->neuron_model->EvaluateTimeDependentEquation(offset2, offset1, AuxNeuronState_p, index, 0);

					float epsi = 1.0f;
					int k = 0;

					//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
					//epsi is the difference between two consecutive aproximation of the implicit method.
					while (epsi > 1e-16 && k < 5){
						this->neuron_model->EvaluateDifferentialEquation(offset2, offset1, AuxNeuronState_p, AuxNeuronState, this->BifixedElapsedTimeInNeuronModelScale);
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_c[j*offset1 + offset2] = Coeficient[state[index] * 4 + 0] * this->BifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j*offset1 + offset2] + Coeficient[state[index] * 4 + 1] * NeuronState[j*SizeStates + index];
							for (int i = 1; i < state[index]; i++){
								AuxNeuronState_c[j*offset1 + offset2] += Coeficient[state[index] * 4 + i + 1] * PreviousNeuronState[(i - 1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
							}
						}

						//jacobian.
						Jacobian(offset2, offset1, AuxNeuronState_p, jacnum);

						for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
							for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
								J[z*offset1*this->neuron_model->N_DifferentialNeuronState + t*offset1 + offset2] = Coeficient[state[index] * 4 + 0] * this->BifixedElapsedTimeInNeuronModelScale * jacnum[z*offset1*this->neuron_model->N_DifferentialNeuronState + t*offset1 + offset2];
								if (z == t){
									J[z*offset1*this->neuron_model->N_DifferentialNeuronState + t*offset1 + offset2] -= 1;
								}
							}
						}
						this->invermat(offset2, offset1, J, inv_J);

						for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
							float aux = 0.0;
							for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
								aux += inv_J[z*offset1*this->neuron_model->N_DifferentialNeuronState + t*offset1 + offset2] * (AuxNeuronState_p[t*offset1 + offset2] - AuxNeuronState_c[t*offset1 + offset2]);
							}
							AuxNeuronState_p1[z*offset1 + offset2] = aux + AuxNeuronState_p[z*offset1 + offset2];
						}

						//We calculate the difference between both aproximations.
						float aux = 0.0;
						float aux2 = 0.0;
						for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
							aux = fabs(AuxNeuronState_p1[z*offset1 + offset2] - AuxNeuronState_p[z*offset1 + offset2]);
							AuxNeuronState_p[z*offset1 + offset2] = AuxNeuronState_p1[z*offset1 + offset2];
							if (aux > aux2){
								aux2 = aux;
							}
						}
						epsi = aux2;
						k++;

					}

					//We increase the state of the integration method.
					if (state[index] < BDForder){
						state[index]++;
					}

					//We acumulate these new values for the next step.
					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						for (int i = (state[index] - 1); i > 0; i--){
							D[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = -D[(i - 1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
						}
						D[0 * SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = AuxNeuronState_p[j*offset1 + offset2] - NeuronState[j*SizeStates + index];
						for (int i = 1; i < state[index]; i++){
							D[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] += D[(i - 1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
						}
					}

					if (state[index]>1){
						for (int i = state[index] - 2; i > 0; i--){
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								PreviousNeuronState[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = PreviousNeuronState[(i - 1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
							}
						}
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							PreviousNeuronState[0 * SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = NeuronState[j*SizeStates + index];
						}
					}
					for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
						NeuronState[z*SizeStates + index] = AuxNeuronState_p[z*offset1 + offset2];
					}

					//Finaly, we evaluate the neural state variables with time dependence.
					this->neuron_model->EvaluateTimeDependentEquation(index, SizeStates, NeuronState, index, 0);

					//Update the last spike time.
					this->neuron_model->vectorNeuronState_GPU2->LastSpikeTimeGPU[index] += this->BifixedElapsedTimeInSeconds;

					this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->BifixedElapsedTimeInNeuronModelScale);
				}
			}
		}


		/*!
		 * \brief It calculate numerically the Jacobian.
		 *
		 * It calculate numerically the Jacobian.
		 *
		 * \param index Index of the cell inside the neuron model.
		 * \param SizeStates Number of neuron.
		 * \param NeuronState Vector of neuron state variables for all neurons.
		 * \param jancum vector where is stored the Jacobian.
		 */
		__device__ void Jacobian(int index, int SizeStates, float * NeuronState, float * jacnum){
			float epsi=9.5367431640625e-7;

			for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				for (int i = 0; i<this->neuron_model->N_NeuronStateVariables; i++){
					AuxNeuronState2[i*SizeStates + index]=NeuronState[i*SizeStates + index];
				}
				AuxNeuronState2[j*SizeStates + index]+=epsi;
				this->neuron_model->EvaluateDifferentialEquation(index, SizeStates, AuxNeuronState2, AuxNeuronState_pos, this->elapsedTimeInNeuronModelScale);

				AuxNeuronState2[j*SizeStates + index]-=2*epsi;
				this->neuron_model->EvaluateDifferentialEquation(index, SizeStates, AuxNeuronState2, AuxNeuronState_neg, this->elapsedTimeInNeuronModelScale);

				for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
					jacnum[z*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = (AuxNeuronState_pos[z*SizeStates + index] - AuxNeuronState_neg[z*SizeStates + index]) / (2 * epsi);
				}

			}
		}

		/*!
		 * \brief It calculate the inverse of a square matrix using Gauss-Jordan Method.
		 *
		 * It calculate the inverse of a square matrix using Gauss-Jordan Method.
		 *
		 * \param index Index of the cell inside the neuron model.
 		 * \param SizeStates Number of neuron.
		 * \param a pointer to the square matrixs.
		 * \param ainv pointer to the inverse of the square matrixs.
		 */
		__device__ void invermat(int index, int SizeStates, float *a, float *ainv) {
			if (this->neuron_model->N_DifferentialNeuronState == 1){
				ainv[0] = 1 / a[0];
			}
			else{
				float coef, inv_elemento;
				int i, j, s;

				for (i = 0; i<this->neuron_model->N_DifferentialNeuronState; i++){
					for (j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						if (i == j)
							ainv[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = 1.0;
						else
							ainv[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = 0.0;
					}
				}
				//Iteraciones
				for (s = 0; s<this->neuron_model->N_DifferentialNeuronState; s++)
				{
					inv_elemento = 1.0 / a[s*SizeStates*this->neuron_model->N_DifferentialNeuronState + s*SizeStates + index];
					for (j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						a[s*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] *= inv_elemento;
						ainv[s*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] *= inv_elemento;
					}

					for (i = 0; i<this->neuron_model->N_DifferentialNeuronState; i++)
					{
						if (i == s)
							;
						else
						{
							coef = a[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + s*SizeStates + index];
							for (j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
								a[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] -= a[s*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] * coef;
								ainv[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] -= ainv[s*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] * coef;
							}

						}
					}
				}
			}
		}


		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param index indicate which neuron must be reseted.
		 *
		 */
		__device__ void resetState(int index){
			state[index]=0;
		}


		/*!
		 * \brief It calculates the conductance exponential values for time driven neuron models.
		 *
		 * It calculates the conductance exponential values for time driven neuron models.
		 */
		__device__ virtual void Calculate_conductance_exp_values(){
			this->neuron_model->Initialize_conductance_exp_values(this->neuron_model->N_TimeDependentNeuronState,1);
			//index 0
			this->neuron_model->Calculate_conductance_exp_values(0, this->BifixedElapsedTimeInNeuronModelScale);
		}
};

#endif /* BDFN_GPU2_H_ */
