
#include ".\include\Utils.h"

int skip_spaces(FILE *fh, long & Currentline){
	int ch;
	while((ch=fgetc(fh)) == ' ' || ch=='\n') // take all spaces
		if(ch=='\n')
			Currentline++;
			
	return(ch);
}

void skip_comments(FILE *fh, long & Currentline){
	int ch;
	while((ch=skip_spaces(fh, Currentline)) == COMMENT_CHAR){
		while((ch=fgetc(fh)) != EOF && ch != '\n');
		
		if(ch=='\n')
			Currentline++;
    }
   
   	if(ch != EOF)
   		ungetc(ch, fh);
}
