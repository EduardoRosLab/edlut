/***************************************************************************
 *                           WithPostSynaptic.h                            *
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

#ifndef WITHPOSTSYNAPTIC_H_
#define WITHPOSTSYNAPTIC_H_

#include "./LearningRule.h"

/*!
 * \file WithPostSynaptic.h
 *
 * \author Francisco Naveros
 * \date November 2013
 *
 * This file declares a class which abstracts a learning rule that implements postsynaptic learning. In this case,
 * each learning rule has just one type of input synapses: normal synapses. When a spike reaches a target neuron
 * through a normal presynaptic connection that implement a learning rule of this type, this connection
 * correlates the spike time with the previous postsynaptic output spike times, thus generating the corresponding LTP or
 * LTD response in function of the learning rule kernel shape. Additionally, when the target neuron generates a
 * postsynaptic spike, this neuron checks the presynaptic activity of all its normal input connections and generates the
 * corresponding LTP or LTD response in each synapse.
 */

/*!
 * \class WithPostSynaptic
 *
 * \brief Learning rule.
 *
 * This class abstract the behaviour of a learning rule that implement postsynaptic learning.
 *
 * \author Francisco Naveros
 * \date November 2013
 */
class WithPostSynaptic : public LearningRule {

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
		WithPostSynaptic();

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~WithPostSynaptic();

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
		 * \brief It returns the default parameters of the learning rule.
		 *
		 * It returns the default parameters of the learning rule. It may be used to obtained the parameters that can be
		 * set for this learning rule.
		 *
		 * \returns A dictionary with the learning rule parameters.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters();

};

#endif /* WITHPOSTSYNAPTIC_H_ */
