#ifndef MULTIPLICATIVEWEIGHTCHANGE_H_
#define MULTIPLICATIVEWEIGHTCHANGE_H_

/*!
 * \file MultiplicativeWeightChange.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a multiplicative learning rule.
 */
 
#include "./WeightChange.h"
 
/*!
 * \class MultiplicativeWeightChange
 *
 * \brief Multiplicative learning rule.
 *
 * This class abstract the behaviour of a multiplicative learning rule.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */ 
class MultiplicativeWeightChange: public WeightChange{
	
	public:
   	
   		/*!
   		 * \brief It applys the weight change function.
   		 * 
   		 * It applys the weight change function.
   		 * 
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyWeightChange(Interconnection * Connection,double SpikeTime);
   		
};


#endif /*MULTIPLICATIVEWEIGHTCHANGE_H_*/
