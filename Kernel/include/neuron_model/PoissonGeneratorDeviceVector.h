/***************************************************************************
 *                           PoissonGeneratorDeviceVector.h                *
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

#ifndef POISSONGENERATORDEVICEVECTOR_H_
#define POISSONGENERATORDEVICEVECTOR_H_

/*!
 * \file PoissonGeneratorDeviceVector.h
 *
 * \author Francisco Naveros
 * \date April 2020
 *
 * This file declares a class which implements a poisson generator device for spike trains. This model does not support input synapses.
 *
 * NOTE: This is a vectorized neuron model. Some configuration parameters has been vectorized in such a way that each neuron in
 * the model can use its own values. The list of vectorized parameters can be obtained with the function "GetVectorizableParameters()"
 */

#include "neuron_model/TimeDrivenInputDevice.h"
#include "simulation/RandomGenerator.h"
#include <string>

using namespace std;


class VectorNeuronState;
class InternalSpike;
class Interconnection;
struct ModelDescription;


/*!
 * \class PoissonGeneratorDeviceVector
 *
 * \brief Poisson generator device for spike trains.This is a vectorized model in which each neuron can take different values in several 
 * configuration parameters.
 *
 *
 * \author Francisco Naveros
 * \date April 2020
 */
class PoissonGeneratorDeviceVector : public TimeDrivenInputDevice {
	protected:
		struct VectorParameters{
			float frequency;
		};

		/*!
		* \brief Structure storing all the neuron model parameters (one copy for each single nueron in the model)
		*/
		VectorParameters * vectorParameters;

		/*!
		 * \brief Poisson generator frequency in Hz units
		 */
		float frequency;

		/*!
		* \brief Constant sampling period in ms units
		*/
		const float sampling_period;

		/*!
		* \brief Pointer to the random generators (one for each "generator")
		*/
		RandomGenerator ** randomGenerator;


	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 */
		PoissonGeneratorDeviceVector();

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~PoissonGeneratorDeviceVector();

		/*!
		 * \brief It return the Neuron Model VectorNeuronState 
		 *
		 * It return the Neuron Model VectorNeuronState 
		 *
		 */
		virtual VectorNeuronState * InitializeState(); 

		
		/*!
		 * \brief Update the neuron state variables.
		 *
		 * It updates the neuron state variables.
		 *
		 * \param index The cell index inside the VectorNeuronState. if index=-1, updating all cell.
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		virtual bool UpdateState(int index, double CurrentTime);

		/*!
		 * \brief It gets the neuron output activity type (spikes or currents).
		 *
		 * It gets the neuron output activity type (spikes or currents).
		 *
		 * \return The neuron output activity type (spikes or currents).
		 */
		enum NeuronModelOutputActivityType GetModelOutputActivityType();

		/*!
		 * \brief It prints the time-driven model info.
		 *
		 * It prints the current time-driven model characteristics.
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
		virtual void InitializeStates(int N_neurons, int OpenMPQueueIndex);

		
		/*!
		 * \brief It returns the neuron model parameters.
		 *
		 * It returns the neuron model parameters.
		 *
		 * \returns A dictionary with the neuron model parameters
		 *
		 * NOTE: this function is accesible throgh the Simulatiob_API interface.
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
		* \brief It returns a vector with the name of the vectorized parameters.
		*
		* It returns a vector with the name of the vectorized parameters.
		*
		* NOTE: this function is accesible throgh the Simulatiob_API interface.
		*/
		static std::map<std::string, std::string> GetVectorizableParameters();

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
		 * \return A newly created NeuronModel object.
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
            if (!TimeDrivenInputDevice::compare(rhs)){
                return false;
            }
            const PoissonGeneratorDeviceVector * e = dynamic_cast<const PoissonGeneratorDeviceVector *> (rhs);
            if (e == 0) return false;

			return this->frequency == e->frequency;
        };

};

#endif /* POISSONGENERATORDEVICEVECTOR_H_ */
