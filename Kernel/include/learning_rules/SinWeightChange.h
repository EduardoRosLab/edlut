/***************************************************************************
 *                           SinWeightChange.h                             *
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

#ifndef SINWEIGHTCHANGE_H_
#define SINWEIGHTCHANGE_H_

/*!
 * \file SinWeightChange.h
 *
 * \author Jesus Garrido
 * \author Niceto Luque
 * \author Richard Carrillo
 * \date July 2009
 *
 * This file declares a class which abstracts a exponential-sinuidal additive learning rule.
 */
 
#include "./AdditiveKernelChange.h"
 
/*!
 * \class SinWeightChange
 *
 * \brief Sinuidal learning rule.
 *
 * This class abstract the behaviour of a exponential-sinusoidal additive learning rule.
 *
 * \author Jesus Garrido
 * \author Niceto Luque
 * \author Richard Carrillo
 * \date July 2009
 */ 
class SinWeightChange: public AdditiveKernelChange{
	private:
	
		/*!
		 * The exponent of the sinusoidal function.
		 */
		int exponent;
		
	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new learning rule.
		 */
		SinWeightChange();

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~SinWeightChange();

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
		 * \brief It gets the number of state variables that this learning rule needs.
		 * 
		 * It gets the number of state variables that this learning rule needs.
		 * 
		 * \return The number of state variables that this learning rule needs.
		 */
   		virtual int GetNumberOfVar() const;
   		
   		/*!
		 * \brief It gets the value of the exponent in the sin function.
		 * 
		 * It gets the value of the exponent in the sin function.
		 * 
		 * \return The value of the exponent in the sin function.
		 */
   		int GetExponent() const;

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
			return "SinAdditiveKernel";
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

#endif /*SINWEIGHTCHANGE_H_*/
