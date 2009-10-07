/***************************************************************************
 *                           TCPIPInputSpikeDriver.h                       *
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

#ifndef TCPIPINPUTSPIKEDRIVER_H_
#define TCPIPINPUTSPIKEDRIVER_H_

/*!
 * \file TCPIPInputSpikeDriver.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class for getting external input spikes from a TCPIP connection.
 */
 
#include "./InputSpikeDriver.h"

#include "../spike/EDLUTFileException.h"

class CdSocket;

/*!
 * \class TCPIPInputSpikeDriver
 *
 * \brief Class for getting input spikes from a TCPIP connection. 
 *
 * This class abstract methods for getting the input spikes to the network. Its subclasses
 * implements the input source and methods.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class TCPIPInputSpikeDriver: public InputSpikeDriver {
	
	private:
	
		/*!
		 * Output spike struct.
		 */
		struct OutputSpike {
			/*!
			 * Number of neuron.
			 */
			long int Neuron;
			
			/*!
			 * Time of the spike.
			 */
			float Time;	
			
			OutputSpike(){};
			
			OutputSpike(int NewNeuron, float NewTime):Neuron(NewNeuron), Time(NewTime){};
		};
	
		/*!
		 * The TCP IP device.
		 */
		CdSocket * Socket;
	
	public:
	
		/*!
		 * \brief Class constructor.
		 * 
		 * It creates a new object from the socket.
		 * 
		 * \param NewSocket The socket connection to send the spikes.
		 * 
		 */
		TCPIPInputSpikeDriver(CdSocket * NewSocket);
		
		/*!
		 * \brief Class desctructor.
		 * 
		 * Class desctructor.
		 */
		~TCPIPInputSpikeDriver();
	
		/*!
		 * \brief It introduces the input activity in the simulation event queue from the connection.
		 * 
		 * This method introduces the cumulated input activity in the simulation event queue.
		 * 
		 * \param Queue The event queue where the input spikes are inserted.
		 * \param Net The network associated to the input spikes.
		 * 
		 * \throw EDLUTException If something wrong happens in the input process.
		 */
		void LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException);
	
};


#endif /*TCPIPINPUTDRIVER_H_*/
