#include "./include/ClientSocket.h"

ClientSocket::ClientSocket(string server_address, unsigned short tcp_port):CdSocket(CLIENT,server_address,tcp_port){
}

ClientSocket::~ClientSocket(){
}