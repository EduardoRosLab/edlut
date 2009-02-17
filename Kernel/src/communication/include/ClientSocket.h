#ifndef CLIENTSOCKET_H_
#define CLIENTSOCKET_H_

#include <string>

#include "./CdSocket.h"

using namespace std;

/*!
 * \brief A socket used to communicate with another program
 *
 * \author Christian Boucheny
 **/
class ClientSocket : public CdSocket {
public:
  /*!
   *
   * \brief Constructor for which parameters of the tcp connection are stored in an external file 
   * 
   * \param server_address address of the server host
   * \param tcp_port tcp_port to connect
   *
   **/
  ClientSocket(string server_address, unsigned short tcp_port);

  /*!
   * \brief Default destructor
   *
   **/
  ~ClientSocket();

};


#endif /*CLIENTSOCKET_H_*/
