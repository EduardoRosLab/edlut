/***************************************************************************
 *                           CdSocket.cpp                                  *
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


#include <iostream>
#include "../../include/communication/CdSocket.h"
#include "../../include/communication/Timer.h"
#include "../../include/communication/CommunicationDevice.h"

#define SERVER_PATH "server"

//( Cd_Socket


CdSocket::CdSocket(enum TCPIPConnectionType status, string server_address,unsigned short tcp_port)
{
//  WORD wVersionRequested = MAKEWORD(1,1);
//  WSADATA wsaData;
//  WSAStartup(wVersionRequested, &wsaData);

  this->status = status;
  this->serv_host_addr = server_address;
  this->serv_tcp_port = tcp_port;

  initializeSocket();
}

CdSocket::~CdSocket()
{
  closesocket(socket_fd);
//  WSACleanup();
}

void CdSocket::initializeSocket()
{  
	/***************/
	/*** CLIENT  ***/
	/***************/
	if(this->status==CLIENT){
	// 	  cout << "Creating INET type client socket" << endl;

		socket_fd = socket(AF_INET,SOCK_STREAM,0);
		if(socket_fd == -1){
			cerr << "Problem when creating the socket: creation ... closing the simulation (CD_SOCKET)" << endl;
			exit(EXIT_FAILURE);
		}      

		struct sockaddr_in serv_addr;
		memset((char *)&serv_addr, 0, sizeof(serv_addr));
		serv_addr.sin_family = AF_INET;      
		serv_addr.sin_addr.s_addr = inet_addr(this->serv_host_addr.c_str());
		serv_addr.sin_port = htons(this->serv_tcp_port);
		int socket_connect = connect(socket_fd,(struct sockaddr *)&serv_addr, sizeof(serv_addr)); 
		if(socket_connect == -1){
			perror("Connect...");
			cerr << "Problem when connecting socket: connection ... closing the simulation ( (CD_SOCKET))" << endl;
			exit(EXIT_FAILURE);
		}
		struct protoent *p;
		int one=1;
		p = getprotobyname("tcp");
		setsockopt(socket_fd, p->p_proto, TCP_NODELAY, (const char*)&one, sizeof(one));
		
		/*int sockfd, n;
    	struct sockaddr_in serv_addr;
    	struct hostent *server;
		
		sockfd = socket(AF_INET, SOCK_STREAM, 0);
	    if (sockfd < 0) 
	        fprintf(stderr, "ERROR opening socket");
	    server = gethostbyname(this->serv_host_addr.c_str());
	    cout << this->serv_host_addr << endl;
	    if (server == NULL) {
	        fprintf(stderr,"ERROR, no such host\n");
	        exit(0);
	    }
	    bzero((char *) &serv_addr, sizeof(serv_addr));
	    serv_addr.sin_family = AF_INET;
	    bcopy((char *)server->h_addr, 
	         (char *)&serv_addr.sin_addr.s_addr,
	         server->h_length);
	    serv_addr.sin_port = htons(this->serv_tcp_port);
	    if (connect(sockfd,(struct sockaddr*)&serv_addr,sizeof(serv_addr)) < 0) 
	        fprintf(stderr, "ERROR connecting");*/  
    }


	/***************/
	/*** SERVER  ***/
	/***************/
	if(this->status==SERVER){
		// 	  cout << "Creating INET socket ..." << endl;

		int tmp_socket_fd, clilen, ret;
		struct sockaddr_in cli_addr, serv_addr;
		memset((char *)&serv_addr, 0, sizeof(serv_addr));

		tmp_socket_fd = socket(AF_INET,SOCK_STREAM,0);       

		serv_addr.sin_family = AF_INET;
		serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
		serv_addr.sin_port = htons(this->serv_tcp_port);

		ret = bind(tmp_socket_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
		if(ret==-1){
			cerr << "Problem when binding socket ... closing the simulation  (CD_SOCKET)" << endl;
			exit(EXIT_FAILURE);
		}
  
		// 	  cout << "Waiting for connection ..." << endl;
		ret = listen(tmp_socket_fd, 1);
		if(ret==-1){
			cerr << "Problem when listening socket ... closing the simulation (CD_SOCKET)" << endl;
			exit(EXIT_FAILURE);
		}
		clilen = sizeof(cli_addr);
		socket_fd = accept(tmp_socket_fd, (struct sockaddr *) &cli_addr, (socklen_t *)&clilen);
		closesocket(tmp_socket_fd);          // close original socket
		if(socket_fd >= 0){
			struct protoent *p;
			int one=1;
			p = getprotobyname("tcp");
			setsockopt(socket_fd, p->p_proto, TCP_NODELAY, (const char*)&one, sizeof(one));  
			// 	      cout << "Connection accepted" << endl;
		}else{
			cerr << "Error (CD_SOCKET): couldn't accept external connection" << endl;
			exit(EXIT_FAILURE);
		}
		
		/*int sockfd = socket(AF_INET, SOCK_STREAM, 0);
		int clilen;
		struct sockaddr_in serv_addr, cli_addr;
	     if (sockfd < 0) 
	        fprintf(stderr, "ERROR opening socket");
	     bzero((char *) &serv_addr, sizeof(serv_addr));
	     serv_addr.sin_family = AF_INET;
	     serv_addr.sin_addr.s_addr = INADDR_ANY;
	     serv_addr.sin_port = htons(this->serv_tcp_port);
	     if (bind(sockfd, (struct sockaddr *) &serv_addr,
	              sizeof(serv_addr)) < 0) 
	              fprintf(stderr, "ERROR on binding");
	     listen(sockfd,5);
	     clilen = sizeof(cli_addr);
	     socket_fd = accept(sockfd, 
	                 (struct sockaddr *) &cli_addr, 
	                 &clilen);
	     if (socket_fd < 0) 
	          fprintf(stderr,"ERROR on accept");*/
	}
}



int CdSocket::receiveBuffer(void* buffer, int buffer_size)
{
  ret_status=0;
  for(int nbytes=0;nbytes<buffer_size;)
    {
      /*** to be sure tha everything is received ***/
      ret_status=recv(socket_fd,((char*)buffer)+nbytes,buffer_size-nbytes,0);
      if(ret_status > 0)
	nbytes+=ret_status;
      else
        {
	  break;
        }
    }
  return(ret_status);
}

int CdSocket::sendBuffer(void* buffer, int buffer_size)
{
  return int(send(socket_fd,(char *)buffer,buffer_size,0));
}

//) Cd_Socket


//) cd_socket.cpp

