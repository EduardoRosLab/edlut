/***************************************************************************
 *                           TimeDrivenNeuronModel_GPU_C_INTERFACE.cuh     *
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

#ifndef TIMEDRIVENNEURONMODEL_GPU_C_INTERFACE_H_
#define TIMEDRIVENNEURONMODEL_GPU_C_INTERFACE_H_

/*!
 * \file TimeDrivenNeuronModel_GPU_C_INTERFACE.cuh
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which abstracts an time-driven neuron model in CPU for GPU.
 */

#include "neuron_model/TimeDrivenModel.h"
#include "neuron_model/CurrentSynapseModel.h"

#include "integration_method/IntegrationMethod_GPU_C_INTERFACE.cuh"
#include "integration_method/IntegrationMethodFactory_GPU_C_INTERFACE.cuh"
#include "simulation/NetworkDescription.h"

#include "neuron_model/TimeDrivenNeuronModel_GPU2.cuh"

#include <string>

using namespace std;

#include "simulation/Configuration.h"
class VectorNeuronState;
class VectorNeuronState_GPU_C_INTERFACE;
class InternalSpike;
class Interconnection;
//struct ModelDescription;


/*!
 * \class TimeDrivenNeuronModel_GPU_C_INTERFACE
 *
 * \brief Time-Driven Spiking neuron model in CPU for GPU
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date November 2012
 */
class TimeDrivenNeuronModel_GPU_C_INTERFACE : public TimeDrivenModel {
	public:


		/*!
		 * \brief Number of CUDA threads.
		*/
		int N_thread;

		/*!
		 * \brief Number of CUDA blocks.
		*/
		int N_block;

		/*!
		 * \brief integration method in CPU for GPU.
		*/
		IntegrationMethod_GPU_C_INTERFACE * integration_method_GPU;

		/*!
		* \brief Pointer to the original VectorNeuronState object with a casting to a VectorNeuronState_GPU_C_INTERFACE object
		*/
		VectorNeuronState_GPU_C_INTERFACE * State_GPU;

		/*!
		 * \brief barrier to synchronize the CPU and the GPU.
		 */
		cudaEvent_t stop;

		/*!
		 * \brief GPU properties
		 */
		cudaDeviceProp prop;

		/*!
		* \brief Object to store the input currents of synapses that receive currents.
		*/
		CurrentSynapseModel * CurrentSynapsis;

		/*!
		* \bried Index identifying in which GPU is executed the neuron model
		*/
		int GPU_index;

		/*!
		* \brief It initializes the CurrentSynapsis object.
		*
		* It initializes the CurrentSynapsis object.
		*
		* \param N_neurons number of neurons.
		*/
		virtual void InitializeCurrentSynapsis(int N_neurons) = 0;

		/*!
		 * \brief Constructor with parameters.
		 *
		 * It generates a new neuron model.
		 *
		 * \param time_scale Variable that indicate which time scale implement this neuron model
		 */
		TimeDrivenNeuronModel_GPU_C_INTERFACE(TimeScale time_scale);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~TimeDrivenNeuronModel_GPU_C_INTERFACE();

		/*!
		* \brief It return the Neuron Model VectorNeuronState
		*
		* It return the Neuron Model VectorNeuronState
		*
		*/
		virtual VectorNeuronState * InitializeState() = 0;


		/*!
		* \brief It processes a propagated spike (input spike in the cell).
		*
		* It processes a propagated spike (input spike in the cell).
		*
		* \note This function doesn't generate the next propagated spike. It must be externally done.
		*
		* \param inter the interconection which propagate the spike
		* \param time the time of the spike.
		*
		* \return A new internal spike if someone is predicted. 0 if none is predicted.
		*/
		virtual InternalSpike * ProcessInputSpike(Interconnection * inter, double time) = 0;

		/*!
		* \brief It processes a propagated current (input current in the cell).
		*
		* It processes a propagated current (input current in the cell).
		*
		* \param inter the interconection which propagate the spike
		* \param target the neuron which receives the spike
		* \param Current input current.
		*/
		virtual void ProcessInputCurrent(Interconnection * inter, Neuron * target, float current) = 0;


		/*!
		 * \brief Update the neuron state variables.
		 *
		 * It updates the neuron state variables.
		 *
		 * \param index The cell index inside the vector. if index=-1, updating all cell.
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		virtual bool UpdateState(int index, double CurrentTime) = 0;


		/*!
		 * \brief It gets the neuron model simulation method (event-driven, time-driven in CPU or time-driven in GPU).
		 *
		 * It gets the neuron model simulation method (event-driven, time-driven in CPU or time-driven in GPU).
		 *
		 * \return The simulation method of the neuron model.
		 */
		enum NeuronModelSimulationMethod GetModelSimulationMethod();

		/*!
		 * \brief It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * \return The neuron model type
		 */
		enum NeuronModelType GetModelType();

		/*!
		 * \brief It gets the neuron output activity type (spikes or currents).
		 *
		 * It gets the neuron output activity type (spikes or currents).
		 *
		 * \return The neuron output activity type (spikes or currents).
		 */
		virtual enum NeuronModelOutputActivityType GetModelOutputActivityType() = 0;

		/*!
		 * \brief It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * \return The neuron input activity types (spikes and/or currents or none).
		 */
		virtual enum NeuronModelInputActivityType GetModelInputActivityType() = 0;


		/*!
		 * \brief It prints the neuron model info.
		 *
		 * It prints the current neuron model characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out) = 0;

		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 * \param OpenMPQueueIndex openmp index
		 */
		virtual void InitializeStates(int N_neurons, int OpenMPQueueIndex)=0;


		/*!
		 * \brief It initialice a neuron model in GPU.
		 *
		 * It initialice a neuron model in GPU.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 */
		virtual void InitializeClassGPU2(int N_neurons)=0;


		/*!
		 * \brief It delete a neuron model in GPU.
		 *
		 * It delete a neuron model in GPU.
		 */
		virtual void DeleteClassGPU2()=0;

		/*!
		 * \brief It create a object of type VectorNeuronState_GPU2 in GPU.
		 *
		 * It create a object of type VectorNeuronState_GPU2 in GPU.
		 */
		virtual void InitializeVectorNeuronState_GPU2()=0;


		/*!
		 * \brief It Checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Type input connection type.
		 *
		 * \return If the neuron model supports this connection type
		 */
		virtual bool CheckSynapseType(Interconnection * connection) = 0;

		/*!
		* \brief It calculates the number of electrical coupling synapses.
		*
		* It calculates the number for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		void CalculateElectricalCouplingSynapseNumber(Interconnection * inter){};

		/*!
		* \brief It allocate memory for electrical coupling synapse dependencies.
		*
		* It allocate memory for electrical coupling synapse dependencies.
		*/
		void InitializeElectricalCouplingSynapseDependencies(){};

		/*!
		* \brief It calculates the dependencies for electrical coupling synapses.
		*
		* It calculates the dependencies for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		void CalculateElectricalCouplingSynapseDependencies(Interconnection * inter){};

		/*!
		* \brief It initialieses the structure required to implement external input current synapsis.
		*
		* It initialieses the structure required to implement external input current synapsis.
		*/
		void InitializeInputCurrentSynapseStructure();

		/*!
		 * \brief It returns the neuron model parameters.
		 *
		 * It returns the neuron model parameters.
		 *
		 * \returns A dictionary with the neuron model parameters
		 */
		virtual std::map<std::string,boost::any> GetParameters() const;

		/*!
		* \brief It returns the neuron model parameters for a specific neuron once the neuron model has been initilized with the number of neurons.
		*
		* It returns the neuron model parameters for a specific neuron once the neuron model has been initilized with the number of neurons.
		*
		* \param index neuron index inside the neuron model.
		*
		* \returns A dictionary with the neuron model parameters
		*
		* NOTE: this function is accesible throgh the Simulatiob_API interface.
		*/
		virtual std::map<std::string, boost::any> GetSpecificNeuronParameters(int index) const noexcept(false) = 0;

		/*!
		 * \brief It loads the neuron model properties.
		 *
		 * It loads the neuron model properties from parameter map.
		 *
		 * \param param_map The dictionary with the neuron model parameters.
		 *
		 * \throw EDLUTException If it happens a mistake with the parameters in the dictionary.
		 */
		virtual void SetParameters(std::map<std::string, boost::any> param_map) noexcept(false);

		/*!
		* \brief Pure virtual function that must be implemented on the derived neuron models creating the integration method
		*
		* It creates the integration methods using the parameter map.
		*
		* \param param_map The dictionary with the integration method parameters.
		*
		* \throw EDLUTException If it happens a mistake with the parameters in the dictionary.
		*/
		virtual IntegrationMethod_GPU_C_INTERFACE * CreateIntegrationMethod(ModelDescription imethodDescription) noexcept(false) = 0;

		/*!
		 * \brief It returns the default parameters of the neuron model.
		 *
		 * It returns the default parameters of the neuron models. It may be used to obtained the parameters that can be
		 * set for this neuron model. This function has been defined as template because it need the neuron model type
		 * of the derived classes.
		 *
		 * \returns A dictionary with the neuron model default parameters.
		 */
		template <class Neuron_Model>
		static std::map<std::string,boost::any> GetDefaultParameters() {
			// Return a dictionary with the parameters
			std::map<std::string,boost::any> newMap = TimeDrivenModel::GetDefaultParameters();

			ModelDescription intMethod;
			intMethod.param_map = IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::GetDefaultParameters_GPU("Euler");
			intMethod.model_name = boost::any_cast<std::string>(intMethod.param_map["name"]);
			newMap["int_meth"] = boost::any(intMethod);
			return newMap;
		}


		/*!
		 * \brief It loads the integration method description.
		 *
		 * It loads the integration method description.
		 *
		 * \param fh Filehandler of the file with the information about the integration method.
		 *
		 * \return An object with the parameters of the neuron model.
		 */
		template <class Neuron_Model>
		static ModelDescription ParseIntegrationMethod(FILE * fh, long & Currentline) noexcept(false){

			char ident_type[MAXIDSIZE + 1];
			skip_comments(fh, Currentline);
			if (fscanf(fh, "%64s", ident_type) != 1) {
				throw EDLUTException(TASK_INTEGRATION_METHOD_TYPE, ERROR_INTEGRATION_METHOD_READ, REPAIR_INTEGRATION_METHOD_READ);
			}
			ModelDescription intMethodDescription = IntegrationMethodFactory_GPU_C_INTERFACE<Neuron_Model>::ParseIntegrationMethod_GPU(ident_type, fh, Currentline);

			return intMethodDescription;
		}

		/*!
		 * \brief Comparison operator between neuron models.
		 *
		 * It compares two neuron models.
		 *
		 * \return True if the neuron models are of the same type and with the same parameters.
		 */
		virtual bool compare(const NeuronModel * rhs) const{
			if (!TimeDrivenModel::compare(rhs)){
				return false;
			}
			const TimeDrivenNeuronModel_GPU_C_INTERFACE * e = dynamic_cast<const TimeDrivenNeuronModel_GPU_C_INTERFACE *> (rhs);
			if (e == 0) return false;

			return this->integration_method_GPU->compare(e->integration_method_GPU);
		};






		void EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){}
		void EvaluateDifferentialEquation(float * NeuronState, float * AuxNeuronState, int index, float elapsed_time){}
		void EvaluateTimeDependentEquation(float * NeuronState, int index, int elapsed_time_index){}
		void Initialize_conductance_exp_values(int N_conductances, int N_elapsed_times){}
		void Calculate_conductance_exp_values(int index, float elapsed_time){}
};

#endif /* TIMEDRIVENNEURONMODEL_GPU_C_INTERFACE_H_ */
