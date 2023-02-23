/***************************************************************************
 *                           BDFn_GPU_C_INTERFACE.cuh                      *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
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

#ifndef BDFN_GPU_C_INTERFACE_H_
#define BDFN_GPU_C_INTERFACE_H_

/*!
 * \file BDFn_GPU_C_INTERFACE.cuh
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implements six fixed step BDF (Backward Differentiation Formulas) integration methods (from
 * first order to sixth order BDF integration method) in GPU (this class is stored in CPU memory and controles the
 * allocation and deleting of GPU auxiliar memory). This method implements a progressive implementation of the
 * higher order integration method using the lower order integration mehtod (BDF1->BDF2->...->BDF6).
 */

#include "../../include/integration_method/IntegrationMethod_GPU_GLOBAL_FUNCTIONS.cuh"
#include "integration_method/FixedStep_GPU_C_INTERFACE.cuh"

#include "../../include/cudaError.h"
//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class BDFn_GPU_C_INTERFACE
 *
 * \brief BDFn integration method in CPU for GPU
 *
 * This class abstracts the initializacion in CPU of six BDF integration methods (BDF1, BDF2, ..., BDF6) for GPU.
 * This CPU class controles the allocation and deleting of GPU auxiliar memory.
 *
 * \author Francisco Naveros
 * \date May 2013
 */
template <class Neuron_Model_GPU>
class BDFn_GPU_C_INTERFACE : public FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>{
	public:

		/*!
		 * \brief These vectors are used as auxiliar vectors in GPU memory.
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
		//For Coeficient
		float * Coeficient;


		/*!
		 * \brief This vector stores previous neuron state variable for all neurons. This one is used as a memory.
		*/
		float * PreviousNeuronState;


		/*!
		 * \brief This vector stores the difference between previous neuron state variable for all neurons. This
		 * one is used as a memory.
		*/
		float * D;

		/*!
		 * \brief This vector contains the state of each neuron (BDF order). When the integration method is reseted (the values of the neuron model variables are
		 * changed outside the integration method, for instance when a neuron spikes and the membrane potential is reseted to the resting potential), the values
		 * store in PreviousNeuronState and D are no longer valid. In this case the order it is set to 0 and must grow in each integration step until it is reache
		 * the target order.
		*/
		int * state;

		/*!
		 * \brief This value stores the order of the integration method.
		*/
		int BDForder;

		/*!
     		* \brief Default constructor of the class.
     		*
     		* It generates a new object.
     		*/
		BDFn_GPU_C_INTERFACE():FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>(), AuxNeuronState(0), AuxNeuronState_p(0),
			AuxNeuronState_p1(0), AuxNeuronState_c(0), jacnum(0), J(0), inv_J(0), AuxNeuronState2(0), AuxNeuronState_pos(0),
			AuxNeuronState_neg(0), Coeficient(0), PreviousNeuronState(0), D(0), state(0){
		}


		/*!
		* \brief Constructor of the class.
		*
		 * It generates a new Euler object.
		*
		* \param NewModel pointer to a template class representing the neuron model.
		*/

		BDFn_GPU_C_INTERFACE(Neuron_Model_GPU *  NewModel):FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>(NewModel), AuxNeuronState(0), AuxNeuronState_p(0),
			AuxNeuronState_p1(0), AuxNeuronState_c(0), jacnum(0), J(0), inv_J(0), AuxNeuronState2(0), AuxNeuronState_pos(0),
			AuxNeuronState_neg(0), Coeficient(0), PreviousNeuronState(0), D(0), state(0){
		}


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~BDFn_GPU_C_INTERFACE(){
			if (AuxNeuronState != 0){
				HANDLE_ERROR(cudaFree(AuxNeuronState));
			}
			if (AuxNeuronState_p != 0){
				HANDLE_ERROR(cudaFree(AuxNeuronState_p));
			}
			if (AuxNeuronState_p1 != 0){
				HANDLE_ERROR(cudaFree(AuxNeuronState_p1));
			}
			if (AuxNeuronState_c != 0){
				HANDLE_ERROR(cudaFree(AuxNeuronState_c));
			}
			if (jacnum != 0){
				HANDLE_ERROR(cudaFree(jacnum));
			}
			if (J != 0){
				HANDLE_ERROR(cudaFree(J));
			}
			if (inv_J != 0){
				HANDLE_ERROR(cudaFree(inv_J));
			}

			if(BDForder>1){
				if (PreviousNeuronState != 0){
					HANDLE_ERROR(cudaFree(PreviousNeuronState));
				}
			}

			if (D != 0){
				HANDLE_ERROR(cudaFree(D));
			}
			if (state != 0){
				HANDLE_ERROR(cudaFree(state));
			}
			if (AuxNeuronState2 != 0){
				HANDLE_ERROR(cudaFree(AuxNeuronState2));
			}
			if (AuxNeuronState_pos != 0){
				HANDLE_ERROR(cudaFree(AuxNeuronState_pos));
			}
			if (AuxNeuronState_neg != 0){
				HANDLE_ERROR(cudaFree(AuxNeuronState_neg));
			}
		}



		/*!
		 * \brief This method reserves all the necesary GPU memory (this memory could be reserved directly in the GPU, but this
		 * suppose some restriction in the amount of memory which can be reserved).
		 *
		 * This method reserves all the necesary GPU memory (this memory could be reserved directly in the GPU, but this
		 * suppose some restriction in the amount of memory which can be reserved).
		 *
		 * \param N_neurons Number of neurons.
		 * \param Total_N_thread Number of thread in GPU.
		 */
		void InitializeMemoryGPU(int N_neurons, int Total_N_thread)
		{
			int size = 16 * sizeof(float *);

			cudaMalloc((void **)&this->Buffer_GPU, size);

			float integration_method_parameters_CPU[2];
			integration_method_parameters_CPU[0] = this->elapsedTimeInSeconds;
			integration_method_parameters_CPU[1] = this->BDForder;
			float * integration_method_parameters_GPU;
			cudaMalloc((void**)&integration_method_parameters_GPU, 2 * sizeof(float));
			cudaMemcpy(integration_method_parameters_GPU, integration_method_parameters_CPU, 2 * sizeof(float), cudaMemcpyHostToDevice);

			cudaMalloc((void**)&AuxNeuronState, this->neuron_model->N_NeuronStateVariables*Total_N_thread*sizeof(float));
			cudaMalloc((void**)&AuxNeuronState_p, this->neuron_model->N_NeuronStateVariables*Total_N_thread*sizeof(float));
			cudaMalloc((void**)&AuxNeuronState_p1, this->neuron_model->N_NeuronStateVariables*Total_N_thread*sizeof(float));
			cudaMalloc((void**)&AuxNeuronState_c, this->neuron_model->N_NeuronStateVariables*Total_N_thread*sizeof(float));
			cudaMalloc((void**)&jacnum, this->neuron_model->N_DifferentialNeuronState*this->neuron_model->N_DifferentialNeuronState*Total_N_thread*sizeof(float));
			cudaMalloc((void**)&J, this->neuron_model->N_DifferentialNeuronState*this->neuron_model->N_DifferentialNeuronState*Total_N_thread*sizeof(float));
			cudaMalloc((void**)&inv_J, this->neuron_model->N_DifferentialNeuronState*this->neuron_model->N_DifferentialNeuronState*Total_N_thread*sizeof(float));

			cudaMalloc((void**)&Coeficient, 7*7*sizeof(float));
			float Coeficient_CPU[7 * 7] = { 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
				1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
				2.0f / 3.0f, 4.0f / 3.0f, -1.0f / 3.0f, 0.0f, 0.0f, 0.0f, 0.0f,
				6.0f / 11.0f, 18.0f / 11.0f, -9.0f / 11.0f, 2.0f / 11.0f, 0.0f, 0.0f, 0.0f,
				12.0f / 25.0f, 48.0f / 25.0f, -36.0f / 25.0f, 16.0f / 25.0f, -3.0f / 25.0f, 0.0f, 0.0f,
				60.0f / 137.0f, 300.0f / 137.0f, -300.0f / 137.0f, 200.0f / 137.0f, -75.0f / 137.0f, 12.0f / 137.0f, 0.0f,
				60.0f / 147.0f, 360.0f / 147.0f, -450.0f / 147.0f, 400.0f / 147.0f, -225.0f / 147.0f, 72.0f / 147.0f, -10.0f / 147.0f };
			cudaMemcpy(Coeficient, Coeficient_CPU, 7*7*sizeof(float), cudaMemcpyHostToDevice);

			if(BDForder>1){
				cudaMalloc((void**)&PreviousNeuronState, (BDForder-1)*N_neurons*this->neuron_model->N_DifferentialNeuronState*sizeof(float));
			}

			cudaMalloc((void**)&D, BDForder*N_neurons*this->neuron_model->N_DifferentialNeuronState*sizeof(float));

			cudaMalloc((void**)&state, N_neurons*sizeof(int));
			cudaMemset(state,0,N_neurons*sizeof(int));

			cudaMalloc((void**)&AuxNeuronState2, this->neuron_model->N_NeuronStateVariables*Total_N_thread*sizeof(float));
			cudaMalloc((void**)&AuxNeuronState_pos, this->neuron_model->N_NeuronStateVariables*Total_N_thread*sizeof(float));
			cudaMalloc((void**)&AuxNeuronState_neg, this->neuron_model->N_NeuronStateVariables*Total_N_thread*sizeof(float));

			Call_BDFn_GPU_C_INTERFACE_memory(this->Buffer_GPU, integration_method_parameters_GPU, AuxNeuronState, AuxNeuronState_p, AuxNeuronState_p1, AuxNeuronState_c, jacnum, J, inv_J, Coeficient, PreviousNeuronState, D, state, AuxNeuronState2, AuxNeuronState_pos, AuxNeuronState_neg);

			cudaFree(integration_method_parameters_GPU);
		}

		/*!
		 * \brief It prints the integration method info.
		 *
		 * It prints the current integration method characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		std::ostream & PrintInfo(std::ostream & out){
			out << "\t\tIntegration Method Type: " << BDFn_GPU_C_INTERFACE::GetName() << endl;
			out << "\t\tBDF order: " << this->BDForder << endl;
			out << "\t\tIntegration Step Time: " << this->elapsedTimeInSeconds<<"s" << endl;

			return out;
		}

		/*!
		 * \brief It returns the integration method parameters.
		 *
		 * It returns the integration method parameters.
		 *
		 * \returns A dictionary with the integration method parameters
		 */
		virtual std::map<std::string, boost::any> GetParameters() const{
			std::map<std::string, boost::any> newMap = FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>::GetParameters();
			newMap["bdf_order"] = this->BDForder;
			newMap["name"] = BDFn_GPU_C_INTERFACE::GetName();
			return newMap;
		}

		/*!
		 * \brief It loads the integration method properties.
		 *
		 * It loads the integration method properties from parameter map.
		 *
		 * \param param_map The dictionary with the integration method parameters.
		 *
		 * \throw EDLUTFileException If it happens a mistake with the parameters in the dictionary.
		 */
		virtual void SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){
			// Search for the parameters in the dictionary
			std::map<std::string, boost::any>::iterator it = param_map.find("bdf_order");
			if (it != param_map.end()){
				int new_BDF_Order = boost::any_cast<int>(it->second);
				if (new_BDF_Order < 1 || new_BDF_Order > 6){
					throw EDLUTException(TASK_BDF_ORDER_LOAD, ERROR_BDF_ORDER_VALUE, REPAIR_BDF_ORDER);
				}
				this->BDForder = new_BDF_Order;
				param_map.erase(it);
			}

			FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>::SetParameters(param_map);
		}

		/*!
		 * \brief It returns the default parameters of the integration method.
		 *
		 * It returns the default parameters of the integration method. It may be used to obtained the parameters that can be
		 * set for this integration method.
		 *
		 * \returns A dictionary with the integration method parameters.
		 */
		static std::map<std::string, boost::any> GetDefaultParameters(){
			std::map<std::string, boost::any> newMap = FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>::GetDefaultParameters();
			newMap["name"] = BDFn_GPU_C_INTERFACE::GetName();
			newMap["bdf_order"] = 2;
			return newMap;
		}

        	/*!
        	 * \brief It loads the integration method description.
        	 *
        	 * It loads the integration method description.
        	 *
        	 * \param fh Filehandler of the file with the information about the integration method.
        	 *
		 * \return An object with the parameters of the integration method.
        	 */
		static ModelDescription ParseIntegrationMethod(FILE * fh, long & Currentline) noexcept(false){
			//Load BDF order.
			skip_comments(fh, Currentline);
			int BDF_Order;
			// Load BDF integration method order
			if (fscanf(fh, "%d", &BDF_Order) != 1) {
				throw EDLUTException(TASK_BDF_ORDER_LOAD, ERROR_BDF_ORDER_READ, REPAIR_BDF_ORDER);
			}
			if (BDF_Order < 1 || BDF_Order > 6){
				throw EDLUTException(TASK_BDF_ORDER_LOAD, ERROR_BDF_ORDER_VALUE, REPAIR_BDF_ORDER);
			}

			//load integration time step.
			ModelDescription nmodel = FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>::ParseIntegrationMethod(fh, Currentline);

			nmodel.param_map["bdf_order"] = BDF_Order;
			nmodel.model_name = BDFn_GPU_C_INTERFACE::GetName();
			return nmodel;
		}

		/*!
		* \brief It returns the name of the integration method
		*
		* It returns the name of the integration method
		*/
		static std::string GetName(){
			return "BDF";
		};

		/*!
		* \brief It creates a new integration method object of this type.
		*
		* It creates a new integration method object of this type.
		*
		* \param param_map The integration method description object.
		*
		* \return A newly created integration method object.
		*/
		static IntegrationMethod_GPU_C_INTERFACE* CreateIntegrationMethod(ModelDescription nmDescription, Neuron_Model_GPU *nmodel){
			BDFn_GPU_C_INTERFACE * newmodel = new BDFn_GPU_C_INTERFACE(nmodel);
			newmodel->SetParameters(nmDescription.param_map);
			return newmodel;
		};

	        /*!
	         * \brief Comparison operator between integration methods.
	         *
	         * It compares two integration methods.
	         *
	         * \return True if the integration methods are of the same type and with the same parameters.
	         */
	        virtual bool compare(const IntegrationMethod_GPU_C_INTERFACE * rhs) const{
			if (!FixedStep_GPU_C_INTERFACE<Neuron_Model_GPU>::compare(rhs)){
				return false;
			}
			const BDFn_GPU_C_INTERFACE *e = dynamic_cast<const BDFn_GPU_C_INTERFACE *> (rhs);
			if (e == 0) return false;
			return this->BDForder == e->BDForder;
	        };


};





#endif /* BDFN_GPU_C_INTERFACE_H_ */
