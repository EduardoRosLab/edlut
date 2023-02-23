/***************************************************************************
 *                           WithTriggerSynaptic.h                         *
 *                           -------------------                           *
 * copyright            : (C) 2023 by Francisco Naveros                    *
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

#ifndef WITHTRIGGERSYNAPTIC_H_
#define WITHTRIGGERSYNAPTIC_H_



/*!
 * \file WithtriggerSynaptic.h
 *
 * \author Francisco Naveros
 * \date February 2023
 *
 * This file declares a class which abstracts a learning rule trigger learning. In this case,
 * each learning rule has two types of input synapses: normal and trigger synapses. When a spike reaches a
 * target neuron through a normal synapse that implement a learning rule of this type, this synapse updates
 * its weight considering this activity and the learning rule parameters (LTP or LTD). By contrast, when the
 * spike is propagated by a trigger connection toward a target neuron, this spike throws another learning
 * mechanism (LTP or LTD) over all the normal input synapses associated to this learning rule considering
 * their past (and in some cases also their future) presynaptic activity.
 * Normal connections are indicated in the network definition using the learning rule index. Trigger connections
 * are indicated in the network definition using a "t" + the learning rule index.
 */

#include "./LearningRule.h"
class Neuron;

/*!
 * \class WithTriggerSynaptic
 *
 * \brief Learning rule.
 *
 * This class abstract the behaviour of a learning rule that does implement trigger learning.
 *
 * \author Francisco Naveros
 * \date February 2023
 */
class WithTriggerSynaptic : public LearningRule {

	public:

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
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new learning rule.
		 */
		WithTriggerSynaptic();

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~WithTriggerSynaptic();

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
		void ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime) {}

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
		bool ImplementPostSynaptic();

		/*!
		 * \brief It returns if this learning rule implements trigger learning.
		 *
		 * It returns if this learning rule implements trigger learning.
		 *
		 * \returns if this learning rule implements trigger learning
		 */
		bool ImplementTriggerSynaptic();

		/*!
		 * \brief It returns the learning rule parameters.
		 *
		 * It returns the learning rule parameters.
		 *
		 * \returns A dictionary with the learning rule parameters
		 */
		virtual std::map<std::string,boost::any> GetParameters();

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

#endif /* WITHTRIGGERSYNAPTIC_H_ */
