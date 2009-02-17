#ifndef SERVERSOCKET_H_
#define SERVERSOCKET_H_

#include "./CdSocket.h"

using namespace std;

/*!
 * \brief A socket used to communicate with another program
 *
 * \author Christian Boucheny
 **/
class ServerSocket : public CdSocket {
public:
  /*!
   *
   * \brief Constructor for which parameters of the tcp connection are stored in an external file 
   * 
   * \param tcp_port tcp_port to connect
   *
   **/
  ServerSocket(unsigned short tcp_port);

  /*!
   * \brief Default destructor
   *
   **/
  ~ServerSocket();

};


#endif /*SERVERSOCKET_H_*/
