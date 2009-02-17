//( communicationdevice.hpp

#ifndef COMMUNICATIONDEVICE_H
#define COMMUNICATIONDEVICE_H


//( CommunicationDevice

/*!
 * \brief Generic object used to communicate with another program
 *
 * \author Christian Boucheny
 **/
class CommunicationDevice 
{
	public:
	  /*!
	   * \brief Default constructor
	   * 
	   **/
	  CommunicationDevice();
	
	  /*!
	   * \brief Default destructor
	   *
	   **/
	  virtual ~CommunicationDevice();
	
	  /*!
	   *
	   * \brief Send a block of data
	   * 
	   * \param   buffer       data to send
	   * \param   buffer_size  size of the packet
	   *
	   * \return  the error signal 
	   *
	   **/
	  virtual int sendBuffer(void* buffer,int buffer_size)=0;
	
	  /*!
	   *
	   * \brief Receive a block of data
	   * 
	   * \param   buffer       data to send
	   * \param   buffer_size  size of the packet
	   *
	   * \return  the error signal 
	   *
	   **/
	  virtual int receiveBuffer(void* buffer,int buffer_size)=0;
	
	protected:

};

//) CommunicationDevice


#endif

//) communicationdevice.hpp
