/***************************************************************************
 *                           WeightChange.h                                *
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

#ifndef WEIGHTCHANGE_H_
#define WEIGHTCHANGE_H_

/*!
 * \file WeightChange.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a learning rule.
 */
 
class Interconnection;
 
/*!
 * \class WeightChange
 *
 * \brief Learning rule.
 *
 * This class abstract the behaviour of a learning rule.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */ 
class WeightChange{
	
	private:
		/*!
		 * Maximum time of the learning rule.
		 */
		float maxpos;
   		
   		/*!
		 * Number of activity registers. 
		 */
   		int numexps;
   		
   		/*!
   		 * This weight change is a trigger.
   		 */
   		int trigger;
   		
   		/*!
   		 * Learning rule parameter 1.
   		 */
   		float a1pre;
   		
   		/*!
   		 * Learning rule parameter 2.
   		 */
   		float a2prepre;
   		
   	public:
   	
   		/*!
		 * \brief It gets the number of state variables that this learning rule needs.
		 * 
		 * It gets the number of state variables that this learning rule needs.
		 * 
		 * \return The number of state variables that this learning rule needs.
		 */
   		virtual int GetNumberOfVar() const = 0;
   	
   		/*!
		 * \brief It gets the maximum time of the learning rule.
		 * 
		 * It gets the maximum time of the learning rule.
		 * 
		 * \return The maximum time of the learning rule.
		 */
   		float GetMaxPos() const;
   		
   		/*!
		 * \brief It sets the maximum time of the learning rule.
		 * 
		 * It sets the maximum time of the learning rule.
		 * 
		 * \param NewMaxPos The maximum time of the learning rule.
		 */
   		void SetMaxPos(float NewMaxPos);
   		
   		/*!
		 * \brief It gets the number of activity registers.
		 * 
		 * It gets the number of activity registers.
		 * 
		 * \return The number of activity registers.
		 */
   		int GetNumExps() const;
   		
   		/*!
		 * \brief It sets the number of activity registers.
		 * 
		 * It sets the number of activity registers.
		 * 
		 * \param NewNumExps The number of activity registers.
		 */
   		void SetNumExps(int NewNumExps);
   		
   		/*!
   		 * \brief It gets a trigger indicator.
   		 * 
   		 * It gets a trigger indicator.
   		 * 
   		 * \return The trigger indicator of the learning rule.
   		 */
   		int GetTrigger() const;
   		
   		/*!
   		 * \brief It sets a trigger indicator.
   		 * 
   		 * It sets a trigger indicator.
   		 * 
   		 * \param NewTrigger The trigger indicator of the learning rule.
   		 */
   		void SetTrigger(int NewTrigger);
   		
   		/*!
   		 * \brief It gets the A1 parameter.
   		 * 
   		 * It gets the A1 parameter.
   		 * 
   		 * \return The A1 parameter.
   		 */
   		float GetA1Pre() const;
   		
   		/*!
   		 * \brief It sets the A1 parameter.
   		 * 
   		 * It sets the A1 parameter.
   		 * 
   		 * \param NewA1Pre The A1 parameter.
   		 */
   		void SetA1Pre(float NewA1Pre);
   		
   		/*!
   		 * \brief It gets the A2 parameter.
   		 * 
   		 * It gets the A2 parameter.
   		 * 
   		 * \return The A2 parameter.
   		 */
   		float GetA2PrePre() const;
   		
   		/*!
   		 * \brief It sets the A2 parameter.
   		 * 
   		 * It sets the A2 parameter.
   		 * 
   		 * \param NewA2PrePre The A2 parameter.
   		 */
   		void SetA2PrePre(float NewA2PrePre);
   		
   		/*!
   		 * \brief It gets the sigma parameter.
   		 * 
   		 * It gets the sigma parameter.
   		 * 
   		 * \return The sigma parameter.
   		 */
   		float GetSigma() const;
   		
   		/*!
   		 * \brief It sets the sigma parameter.
   		 * 
   		 * It sets the sigma parameter.
   		 * 
   		 * \param NewSigma The sigma parameter.
   		 */
   		void SetSigma(float NewSigma);
   		
   		/*!
   		 * \brief It applys the weight change function.
   		 * 
   		 * It applys the weight change function.
   		 * 
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyWeightChange(Interconnection * Connection,double SpikeTime) = 0;
   		
};

#endif /*WEIGHTCHANGE_H_*/
