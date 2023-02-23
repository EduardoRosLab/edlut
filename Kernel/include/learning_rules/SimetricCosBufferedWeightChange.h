/***************************************************************************
 *                           SimetricCosBufferedWeightChange.h             *
 *                           -------------------                           *
 * copyright            : (C) 2016 by Francisco Naveros and Niceto Luque   *
 * email                : fnaveros@ugr.es nluque@ugr.es                    *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef SIMETRICCOSBUFFEREDWEIGHTCHANGE_H_
#define SIMETRICCOSBUFFEREDWEIGHTCHANGE_H_

/*!
 * \file SimetricCosBufferedWeightChange.h
 *
 * \author Francisco Naveros
 * \author Niceto Luque
 * \date May 2015
 *
 * This file declares a class which abstracts the behaviour of a exponential-cosinusoidal additive learning rule. When a spike arrive 
 * for a non-trigger synapse, a LTP method with value a1pre is applied. When a spike arrive for a trigger synapse, a LTD method with kernel 
 * a2prepre*exp(exponent*t/tau)*cos^2((pi/2)*t/tau), for all non-trigger synapses. This kernel is applied to the previous and future activity
 * to the trigger synapse. The kernel is precomputed in a look-up table.
 */
 
#include "./WithTriggerSynaptic.h"
#include "../simulation/NetworkDescription.h"

class BufferedActivityTimes;
 
/*!
 * \class SimetricCosBufferedWeightChange
 *
 * \brief Cosinusoidal learning rule.
 *
  * This class abstract the behaviour of a exponential-cosinusoidal additive learning rule. When a spike arrive for a non-trigger synapse,
 * a LTP method with value a1pre is applied. When a spike arrive for a trigger synapse, a LTD method with kernel 
 * a2prepre*exp(exponent*t/tau)*cos^2((pi/2)*t/tau), for all non-trigger synapses. This kernel is applied only to the previous  and future
 * activity to the trigger synapse. The kernel is precomputed in a look-up table.
 *
 * \author Francisco Naveros
 * \author Niceto Luque
 * \date May 2016
 */ 

class SimetricCosBufferedWeightChange: public WithTriggerSynaptic{
	private:


		/*!
		 * \brief Kernel amplitude in second.
		 */
		float tau;

		/*!
		 * \brief Exponent
		 */
		float exponent;

		/*!
		 * \brief Maximum weight change for LTP
		 */
		float fixwchange;

		/*!
		 * \brief Maximum weight change LTD
		 */
		float kernelwchange;

		/*!
		* Maximum time calculated in the look-up table.
		*/
		double maxTimeMeasured;
		double inv_maxTimeMeasured;

		/*!
		* Number of elements inside the look-up table.
		*/
		int N_elements;

		/*!
		* Look-up table for the kernel.
		*/
		float * kernelLookupTable;

		/*!
		* Buffer of spikes propagated by "no trigger" synapses
		*/
		BufferedActivityTimes * bufferedActivityTimesNoTrigger;

		/*!
		* Buffer of spikes propagated by "trigger" synapses
		*/
		BufferedActivityTimes * bufferedActivityTimesTrigger;

	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new learning rule.
		 */
		SimetricCosBufferedWeightChange();

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~SimetricCosBufferedWeightChange();

		/*!
		 * \brief It initialize the state associated to the learning rule for all the synapses.
		 *
		 * It initialize the state associated to the learning rule for all the synapses.
		 *
		 * \param NumberOfSynapses the number of synapses that implement this learning rule.
		 * \param NumberOfNeurons the total number of neurons in the network
		 */
		void InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons);


		/*!
		 * \brief It loads the learning rule properties.
		 *
		 * It loads the learning rule properties.
		 *
		 * \param fh A file handler placed where the Learning rule properties are defined.
		 *
		 * \return The learning rule description object.
		 *
		 * \throw EDLUTException If something wrong happens in reading the learning rule properties.
		 */
		static ModelDescription ParseLearningRule(FILE * fh) noexcept(false);

		/*!
   		 * \brief It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime);

   		/*!
		 * \brief It prints the learning rule info.
		 *
		 * It prints the current learning rule characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out);

		/*!
		 * \brief It creates a new learning rule object of this type.
		 *
		 * It creates a new learning rule object of this type.
		 *
		 * \param param_map The learning rule description object.
		 *
		 * \return A newly created ExpWeightChange object.
		 */
		static LearningRule* CreateLearningRule(ModelDescription lrDescription);

		/*!
		 * \brief It provides the name of the learning rule
		 *
		 * It provides the name of the learning rule, i.e. the name that can be mentioned to use this learning rule.
		 *
		 * \return The name of the learning rule
		 */
		static std::string GetName(){
			return "SimetricCosBufferedAdditiveKernel";
		};

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

#endif /*SIMETRICCOSBUFFEREDWEIGHTCHANGE_H_*/
