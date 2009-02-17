#include "./include/ServerSocket.h"

ServerSocket::ServerSocket(unsigned short tcp_port):CdSocket(SERVER,"",tcp_port){
}

ServerSocket::~ServerSocket(){
}