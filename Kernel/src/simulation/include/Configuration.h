#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

/*!
 * \file Configuration.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file defines some configuration constants for simulation.
 */
 
//#define min(a,b) (((a)<(b))?(a):(b))

/*!
 * Time of the next spike when no spike is predicted.
 */
#define NOPREDICTION -1.0

/*!
 * Multiplicative factor of synaptic weights.
 */
#define WEIGHTSCALE 1

/*!
 * Maximum number of information lines.
 */
#define MAXINFOLINES 5

/*!
 * Default refractary period
 */
#define DEF_REF_PERIOD 2e-3

/*!
 * Maximum number of state variables.
 */
#define MAXSTATEVARS 7

/*!
 * Maximum number of characters of a type identificator.
 */
#define MAXIDSIZE 32

/*!
 * Maximum number of characters of a type identificator.
 */
#define MAXIDSIZEC "32"

#endif /*CONFIGURATION_H_*/
