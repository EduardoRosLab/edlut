/***************************************************************************
 *                           NeuronModel.h                                 *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@ugr.es             *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef NEURONMODEL_H_
#define NEURONMODEL_H_

/*!
 * \file NeuronModel.h
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date January 2011
 *
 * \note Modified on January 2012 in order to include time-driven simulation support in GPU.
 * New state variables (TIME_DRIVEN_MODEL_CPU and TIME_DRIVEN_MODEL_GPU)
 *
 * \note Modified on June 2018 to add online parameter setting.
 *
 * This file declares a class which abstracts an spiking neural model.
 */

#include <string>
#include <string.h>

#include "spike/EDLUTFileException.h"
//#include "../../include/simulation/ExponentialTable.h"
//#include "../../include/integration_method/IntegrationMethod.h"

#include <map>
#include <boost/any.hpp>

using namespace std;

class VectorNeuronState;
class InternalSpike;
class Interconnection;
class Neuron;
class NeuronModelPropagationDelayStructure;


//This variable indicates if the neuron model is an event driven neuron model (ASYNCHRONOUS UPDATE) or a time driven neuron model in CPU or GPU (SYNCHRONOUS UPDATE).
enum NeuronModelSimulationMethod { EVENT_DRIVEN_MODEL, TIME_DRIVEN_MODEL_CPU, TIME_DRIVEN_MODEL_GPU};

//This variable indicates if the neuron model is an input device that inyects activity (spikes or currents) in the neural network or is a neuron layer.
enum NeuronModelType { INPUT_DEVICE, NEURAL_LAYER };

//This variable indicates if the neuron model generates output spikes or currents.
enum NeuronModelOutputActivityType { OUTPUT_SPIKE, OUTPUT_CURRENT};

//This variable indicates if the neuron model can receive input spikes and/or currents or none (INPUT_DEVICES do not receive input synapses).
enum NeuronModelInputActivityType { INPUT_SPIKE, INPUT_CURRENT, INPUT_SPIKE_AND_CURRENT, NONE_INPUT};





//This variable indicate if a neuron model is defined in a second or milisicond time scale.
enum TimeScale {SecondScale=1, MilisecondScale=1000};


/*!
 * \class NeuronModel
 *
 * \brief Spiking neuron model
 *
 * This class abstracts the behavior of a neuron in a spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Jesus Garrido
 * \date January 2011
 */
class NeuronModel {

    private:

        /*!
         * \brief Neuron model name
         */
        std::string name;

	protected:

		/*!
		 * \brief This variable indicate if the neuron model has a time scale of seconds (1) or miliseconds (1000).
		*/
		float time_scale;
		float inv_time_scale;


	public:

		/*!
		* \brief Initial state of this neuron model
		*/
		VectorNeuronState * State;


		/*!
		 * \brief PropagationStructure Object that include a structure of all the propagation delay of the neuron that composse this neuron model.
		 */
		NeuronModelPropagationDelayStructure * PropagationStructure;

		/*!
		* \brief Default constructor without parameters.
		*
		* It generates a new neuron model object without being initialized.
		*/
		NeuronModel();
		
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 */
		NeuronModel(TimeScale new_time_scale);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~NeuronModel();

		/*!
		 * \brief It initializes the neuron state to defined values.
		 *
		 * It initializes the neuron state to defined values.
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
		 * \brief It gets the neuron model simulation method (event-driven, time-driven in CPU or time-driven in GPU).
		 *
		 * It gets the neuron model simulation method (event-driven, time-driven in CPU or time-driven in GPU).
		 *
		 * \return The simulation method of the neuron model.
		 */
		virtual enum NeuronModelSimulationMethod GetModelSimulationMethod() = 0;

		/*!
		 * \brief It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * \return The neuron model type
		 */
		virtual enum NeuronModelType GetModelType() = 0;

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
		 * \brief It gets the VectorNeuronState.
		 *
		 * It gets the VectorNeuronState.
		 *
		 * \return The VectorNeuronState.
		 */
		//VectorNeuronState * GetVectorNeuronState();
		inline VectorNeuronState * GetVectorNeuronState(){
			return this->State;
		}

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
		 * \brief It checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Type input connection type.
		 *
		 * \return If the neuron model supports this connection type
		 */
		virtual bool CheckSynapseType(Interconnection * connection)=0;


		/*!
		 * \brief It returns the NeuronModelPropagationDelayStructure object.
		 *
		 * It returns the NeuronModelPropagationDelayStructure object.
		 *
		 * \return the NeuronModelPropagationDelayStructure object.
		 */
		NeuronModelPropagationDelayStructure * GetNeuronModelPropagationDelayStructure();


		/*!
		 * \brief It sets the neuron model time scale.
		 *
		 * It sets the neuron model time scale.
		 *
		 * \param new_time_scale time scale.
		 */
		void SetTimeScale(float new_time_scale);


		/*!
		 * \brief It gets the neuron model time scale.
		 *
		 * It gets the neuron model time scale.
		 *
		 * \return time scale.
		 */
		float GetTimeScale() const;



		/*!
		* \brief It calculates the number of electrical coupling synapses.
		*
		* It calculates the number for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		virtual void CalculateElectricalCouplingSynapseNumber(Interconnection * inter)=0;

		/*!
		* \brief It allocate memory for electrical coupling synapse dependencies.
		*
		* It allocate memory for electrical coupling synapse dependencies.
		*/
		virtual void InitializeElectricalCouplingSynapseDependencies()=0;

		/*!
		* \brief It calculates the dependencies for electrical coupling synapses.
		*
		* It calculates the dependencies for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		virtual void CalculateElectricalCouplingSynapseDependencies(Interconnection * inter) = 0;
		
		
		/*!
		* \brief It initialieses the structure required to implement external input current synapsis.
		*
		* It initialieses the structure required to implement external input current synapsis.
		*/
		virtual void InitializeInputCurrentSynapseStructure() = 0;

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
		* \brief It loads the neuron model properties for a specific neuron once the neuron model has been initilized with the number of neurons.
		*
		* It loads the neuron model properties from parameter map for a specific neuron once the neuron model has been initilized with the number of neurons.
		*
		* \param index neuron index inside the neuron model.
		* \param param_map The dictionary with the neuron model parameters.
		*
		* \throw EDLUTException If it happens a mistake with the parameters in the dictionary.
		*
		* NOTE: this function is accesible throgh the Simulatiob_API interface.
		*/
		virtual void SetSpecificNeuronParameters(int index, std::map<std::string, boost::any> param_map) noexcept(false);

		/*!
		 * \brief It returns the default parameters of the neuron model.
		 *
		 * It returns the default parameters of the neuron models. It may be used to obtained the parameters that can be
		 * set for this neuron model.
		 *
		 * \returns A dictionary with the neuron model default parameters.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters();

        /*!
         * \brief It returns the neuron model name.
         *
         * It returns the neuron model name.
         *
         * \returns The neuron model name.
         */
        std::string GetNeuronModelName() const;

        /*!
         * \brief Comparison operator between neuron models.
         *
         * It compares two neuron models.
         *
         * \return True if the neuron models are of the same type and with the same parameters.
         */
         virtual bool compare(const NeuronModel * rhs) const{
             return this->name==rhs->name && this->time_scale==rhs->time_scale;
         };
};

#endif /* NEURONMODEL_H_ */
