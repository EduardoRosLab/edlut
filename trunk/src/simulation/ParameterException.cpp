#include ".\include\ParameterException.h"

ParameterException::ParameterException(string ErrorParameter, string ErrorMessage):Parameter(ErrorParameter), Message(ErrorMessage){
}
		
string ParameterException::GetParameter() const{
	return this->Parameter;
}
		
string ParameterException::GetErrorMsg() const{
	return this->Message;
}
		
void ParameterException::display_error() const{
	cerr << "Invalid parameter " << this->Parameter << endl;
	cerr << "Error message: " << this->Message << endl;
}

ostream & operator<< (ostream & out, ParameterException Exception){
	out << "Invalid parameter " << Exception.GetParameter() << endl;
	out << "Error message: " << Exception.GetErrorMsg() << endl;
	return out;	
}