/***************************************************************************
 *                           LearningRule.h                                *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#ifndef LEARNINGRULE_H_
#define LEARNINGRULE_H_

#include "../simulation/PrintableObject.h"

#include "../spike/EDLUTFileException.h"

#include <boost/any.hpp>


/*!
 * \file LearningRule.h
 *
 * \author Jesus Garrido
 * \date August 2010
 *
 * This class abstracts the behavior of a learning rule in a spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, etc.).
 * This is only a virtual class (an interface) which defines the functions of the
 * inherited classes.
 */

#include "../../include/learning_rules/ConnectionState.h"
#include <map>
#include <boost/any.hpp>

class Interconnection;
class Neuron;


/*!
 * \class LearningRule
 *
 * \brief Learning rule.
 *
 * This class abstracts the behavior of a learning rule in a spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, etc.).
 * This is only a virtual class (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Jesus Garrido
 * \date March 2010
 */
class LearningRule : public PrintableObject {
	private:
		/*!
		 * \brief The index of the learning rule in the network
		 */
		unsigned int learningRuleIndex;


	public:

		/*!
		 * \brief The conection state of the learning rule.
		 */
		ConnectionState * State;

		/*!
		 * \brief An auxiliar variable to manage the asignation of index.
		 */
		int counter;

		/*!
		 * \brief It initialize the state associated to the learning rule for all the synapses.
		 *
		 * It initialize the state associated to the learning rule for all the synapses.
		 *
		 * \param NumberOfSynapses the number of synapses that implement this learning rule.
		 * \param NumberOfNeurons the total number of neurons in the network
		 */
		virtual void InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons) = 0;


		/*!
		 * \brief It return the state associated to the learning rule for all the synapses.
		 *
		 * It return the state associated to the learning rule for all the synapses.
		 *
		 * \return the learning rule state for all the synapses.
		 */
		ConnectionState * GetConnectionState();

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new learning rule.
		 *
		 */
		LearningRule();

		/*!
		 * \brief Object destructor.
		 *
		 * It remove a LearningRule object an releases the memory of the ConnectionState.
		 */
		virtual ~LearningRule();

		/*!
		 * \brief Return the index of the learning rule in the network
		 *
		 * Return the index of the learning rule in the network
		 */
		inline unsigned int GetLearningRuleIndex(){
			return this->learningRuleIndex;
		}

		/*!
		 * \brief Set the index of the learning rule in the network
		 *
		 * Set the index of the learning rule in the network
		 *
		 * \param NewLearningRuleIndex The index of the learning rule in the network
		 */
		inline void SetLearningRuleIndex(unsigned int NewLearningRuleIndex){
			this->learningRuleIndex = NewLearningRuleIndex;
		}


        /*!
   		 * \brief It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime) = 0;

		/*!
		* \brief It applies the weight change function to all its input synapses when a postsynaptic spike arrives.
		*
		* It applies the weight change function to all its input synapses when a postsynaptic spike arrives.
		*
		* \param neuron The target neuron that manage the postsynaptic spike
		* \param SpikeTime The spike time of the postsynaptic spike.
		*/
		virtual void ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime) = 0;

   		/*!
		 * \brief It prints the learning rule info.
		 *
		 * It prints the current learning rule characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out) = 0;

   	/*!
		 * \brief It returns if this learning rule implements postsynaptic learning.
		 *
		 * It returns if this learning rule implements postsynaptic learning.
		 *
		 * \returns if this learning rule implements postsynaptic learning
		 */
		virtual bool ImplementPostSynaptic() = 0;

		/*!
		 * \brief It returns if this learning rule implements trigger learning.
		 *
		 * It returns if this learning rule implements trigger learning.
		 *
		 * \returns if this learning rule implements trigger learning
		 */
		virtual bool ImplementTriggerSynaptic() = 0;

		/*!
		 * \brief It returns the learning rule parameters.
		 *
		 * It returns the learning rule parameters.
		 *
		 * \returns A dictionary with the learning rule parameters
		 */
		virtual std::map<std::string,boost::any> GetParameters();

		/*!
		 * \brief It loads the learning rule properties.
		 *
		 * It loads the learning rule properties from parameter map.
		 *
		 * \param param_map The dictionary with the learning rule parameters.
		 *
		 * \throw EDLUTFileException If it happens a mistake with the parameters in the dictionary.
		 */
		virtual void SetParameters(std::map<std::string, boost::any> param_map) noexcept(false);

		/*!
		 * \brief It returns the default parameters of the learning rule.
		 *
		 * It returns the default parameters of the learning rule. It may be used to obtained the parameters that can be
		 * set for this learning rule.
		 *
		 * \returns A dictionary with the learning rule parameters.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters();


};

#endif /* LEARNINGRULE_H_ */
