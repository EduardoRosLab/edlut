/***************************************************************************
 *                           ExpWeightChange.h                             *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
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

#ifndef EXPWEIGHTCHANGE_H_
#define EXPWEIGHTCHANGE_H_

/*!
 * \file ExpWeightChange.h
 *
 * \author Jesus Garrido
 * \date October 2011
 *
 * This file declares a class which abstracts a exponential additive learning rule.
 */
 
#include "./AdditiveKernelChange.h"
 
/*!
 * \class ExpWeightChange
 *
 * \brief Exponential learning rule.
 *
 * This class abstract the behaviour of a exponential additive learning rule.
 *
 * \author Jesus Garrido
 * \date October 2011
 */ 
class ExpWeightChange: public AdditiveKernelChange{
	private:
		

	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new learning rule with its index.
		 */
		ExpWeightChange();

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~ExpWeightChange();

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
			return "ExpAdditiveKernel";
		};

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

#endif /*EXPWEIGHTCHANGE_H_*/
