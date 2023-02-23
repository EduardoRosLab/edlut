/***************************************************************************
 *                           AdditiveKernelChange.h                        *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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

#ifndef ADDITIVEKERNELCHANGE_H_
#define ADDITIVEKERNELCHANGE_H_

/*!
 * \file AdditiveKernelChange.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date March 2010
 *
 * This file declares a class which abstracts an additive learning rule.
 */

#include "../../include/learning_rules/WithTriggerSynaptic.h"
#include "../../include/simulation/NetworkDescription.h"
/*!
 * \class AdditiveKernelChange
 *
 * \brief Additive learning rule with kernel.
 *
 * This class abstract the behaviour of a additive learning rule with kernel.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date March 2010
 */
class AdditiveKernelChange : public WithTriggerSynaptic {
	protected:
		/*!
		 * Maximum time of the learning rule.
		 */
		float kernelpeak;

		/*!
		 * Learning rule parameter 1.
		 */
		float fixwchange;

		/*!
		 * Learning rule parameter 2.
		 */
		float kernelwchange;

	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new learning rule.
		 */
		AdditiveKernelChange();


		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~AdditiveKernelChange();


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
		virtual ostream & PrintInfo(ostream & out) = 0;

		/*!
		 * \brief It gets the number of state variables that this learning rule needs.
		 *
		 * It gets the number of state variables that this learning rule needs.
		 *
		 * \return The number of state variables that this learning rule needs.
		 */
		virtual int GetNumberOfVar() const;

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



#endif /* ADDITIVEKERNELCHANGE_H_ */
