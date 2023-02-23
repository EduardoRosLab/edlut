/***************************************************************************
 *                           InputSpikeNeuronModel.h                       *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#ifndef INPUTSPIKENEURONMODEL_H_
#define INPUTSPIKENEURONMODEL_H_

/*!
 * \file InputSpikeNeuronModel.h
 *
 * \author Francisco Naveros
 * \date July 2015
 *
 * This file declares a neuron model that can emulate an input layer of neurons that propagates InputSpike events.
 */

#include "neuron_model/EventDrivenInputDevice.h"

//#include "../spike/EDLUTFileException.h"

//#include <iostream>

using namespace std;

//class Neuron;

class VectorNeuronState;
class InternalSpike;
//class Interconnection;
struct ModelDescription;


/*!
 * \class InputSpikeNeuronModel
 *
 * \brief Input neuron model
 *
 * This class defines the behavior of an input neuron layer that can propagate input spikes to the neural network. It includes 
 * internal model functions which define the behavior of the model.
 *
 * \author Francisco Naveros
 * \date July 2015
 */
class InputSpikeNeuronModel : public EventDrivenInputDevice {
	public:


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 */
		InputSpikeNeuronModel();

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~InputSpikeNeuronModel();


		/*!
		 * \brief It initializes the neuron state to defined values.
		 *
		 * It initializes the neuron state to defined values.
		 *
		 */
		VectorNeuronState * InitializeState() {
			return NULL;
		};

		/*!
		* \brief It gets the neuron model generator type (spike or current).
		*
		* It gets the neuron model generator type (spike or current).
		*
		* \return The neuron model generator type (spike or current).
		*/
		enum NeuronModelOutputActivityType GetModelOutputActivityType();



		/*!
		 * \brief It prints the neuron model info.
		 *
		 * It prints the current neuron model characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out);

		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 * \param OpenMPQueueIndex openmp index
		 */
		void InitializeStates(int N_neurons, int OpenMPQueueIndex){
		
		};
		
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
		virtual std::map<std::string, boost::any> GetSpecificNeuronParameters(int index) const noexcept(false);

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
		 * \brief It returns the default parameters of the neuron model.
		 *
		 * It returns the default parameters of the neuron models. It may be used to obtained the parameters that can be
		 * set for this neuron model.
		 *
		 * \returns A dictionary with the neuron model default parameters.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters();

		/*!
		 * \brief It creates a new neuron model object of this type.
		 *
		 * It creates a new neuron model object of this type.
		 *
		 * \param param_map The neuron model description object.
		 *
		 * \return A newly created InputNeuronModel object.
		 */
		static NeuronModel* CreateNeuronModel(ModelDescription nmDescription);

		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 *
		 * \param FileName This parameter is not used. It is stub parameter for homegeneity with other neuron models.
		 *
		 * \return A neuron model description object with the parameters of the neuron model.
		 */
		static ModelDescription ParseNeuronModel(std::string FileName) noexcept(false);

		/*!
		 * \brief It returns the name of the neuron type
		 *
		 * It returns the name of the neuron type.
		 */
		static std::string GetName();

		/*!
		* \brief It returns the neuron model information, including its parameters.
		*
		* It returns the neuron model information, including its parameters.
		*
		*\return a map with the neuron model information, including its parameters.
		*/
		static std::map<std::string, std::string> GetNeuronModelInfo();

        /*!
         * \brief Comparison operator between neuron models.
         *
         * It compares two neuron models.
         *
         * \return True if the neuron models are of the same type and with the same parameters.
         */
        virtual bool compare(const NeuronModel * rhs) const{
            if (!EventDrivenInputDevice::compare(rhs)){
                return false;
            }
            const InputSpikeNeuronModel * e = dynamic_cast<const InputSpikeNeuronModel *> (rhs);
            if (e == 0) return false;

            return true;
        };
};

#endif /* EVENTDRIVENNEURONMODEL_H_ */
