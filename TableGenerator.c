// Version 2.1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define MAXIDSIZE 32
#define MAXIDSIZEC "32"
#define MAXINTERVALS 6
#define MAXFILES 7
#define MAXTABLES 7
// the table configuration file may want to access Tables structure

struct tcoordinates
  {
   float **ranges;
   int **neqsel;
   unsigned long *size;
   unsigned long *posdim;
  };

struct ttable
  {
   float *elems;
   struct tcoordinates coord;
  };

// Generated-table global variable
struct ttable Tables[MAXTABLES];

#include "tab2cfg.c"
#define MAXDIMENSIONS (NUM_EQS+1)
#define NEQSYS sizeof(Eq_sys)/((sizeof(void*)+sizeof(int))*NUM_EQS)

struct tdimensioncfg
  {
   int var;
   long intervals;
   struct
     {
      double c0,ci,cf,log;
      unsigned long nco;
      int neqsel;
     } interval[MAXINTERVALS];
  };

struct ttablecfg
  {
   float varinit[NUM_EQS];
   int ndiffeqs[NEQSYS];
   int diffeqs[NEQSYS][NUM_EQS];
   int nothereqs[NEQSYS];
   int othereqs[NEQSYS][NUM_EQS];
   int outputvar;
   unsigned long ndimensions;
   struct tdimensioncfg dimension[MAXDIMENSIONS];
  };

struct tconfdef
  {
   int nfiles;
   struct
     {
      char ident[MAXIDSIZE+1];
      int ntables;
      struct consts cons;
      struct ttablecfg table[MAXTABLES];
     } file[MAXFILES];
  };

// Table definition global variable
struct tconfdef Confdef;
int Currentfile;

// Variabe update global variable
int Varupdate[NUM_EQS];

// Global statistical variables
unsigned long Function_evaluations;
unsigned long Numeric_errs;

// Global parameters
float Error_tolerance=0.002;
float Time_step_folding=1.2;
float Max_retries=-1; //-1 = infinite (modify if approx never ends)

long Currentline;
#define COMMENT_CHAR '/'
int skip_spaces(FILE *fh)
  {
   int ch;
   while((ch=fgetc(fh)) == ' ' || ch=='\n') // take all spaces
      if(ch=='\n')
         Currentline++;
   return(ch);
  }

int skip_comments(FILE *fh)
  {
   int ch;
   while((ch=skip_spaces(fh)) == COMMENT_CHAR)
     {
      while((ch=fgetc(fh)) != EOF && ch != '\n');
      if(ch=='\n')
         Currentline++;
     }
   if(ch != EOF)
      ungetc(ch, fh);
   return(1);
  }

int find_parameters(FILE *fh)
  {
   char id_str[]="$PAR$";
   int id_recog;
   int ch;
   id_recog=0;
   while(id_recog < sizeof(id_str)-1)
     {
      ch=fgetc(fh);
      if(ch == EOF)
         break;
      else
         if(ch == id_str[id_recog])
            id_recog++;
         else
           {
            id_recog=0;
            if(ch == '\n')
               Currentline++;
           }
     }
   return(id_recog == sizeof(id_str)-1);
  }

void lineerror(int errwar, char *msg)
  {
   printf("%s reading data from configuration file\nIn line: %li\n%s\n",(errwar)?"Error":"Warning",Currentline,msg);
  }

inline float table_access(int ntable, union vars *v)
  {
   static unsigned int coo[MAXTABLES][MAXDIMENSIONS]={{0}};
   unsigned long ndim,tabpos;
   struct ttablecfg *ctableconf;
   struct ttable *ctableel;
   unsigned int dvar;
   ctableconf=&Confdef.file[Currentfile].table[ntable];
   ctableel=&Tables[ntable];
   tabpos=0;
   for(ndim=0;ndim<ctableconf->ndimensions;ndim++)
     {
      dvar=ctableconf->dimension[ndim].var;
      if(coo[ntable][ndim] >= ctableel->coord.size[ndim])
         coo[ntable][ndim]=ctableel->coord.size[ndim]-1;
         
      for(;coo[ntable][ndim]>0 && ctableel->coord.ranges[ndim][coo[ntable][ndim]] > v->list[dvar];coo[ntable][ndim]--); // search for the coordinate backward
      for(;coo[ntable][ndim]<ctableel->coord.size[ndim]-1 && ctableel->coord.ranges[ndim][coo[ntable][ndim]] < v->list[dvar];coo[ntable][ndim]++); // search forward
      tabpos+=coo[ntable][ndim]*ctableel->coord.posdim[ndim]; // calculate input-table position
     }
   return(ctableel->elems[tabpos]);
  }

int load_dimconf(FILE *fh, struct ttablecfg *tab)
  {
   int ret;
   ret=1;
   skip_comments(fh);
   if(fscanf(fh,"%lu",&tab->ndimensions)==1)
     {
      if(tab->ndimensions <= MAXDIMENSIONS && tab->ndimensions > 0)
        {
         struct tdimensioncfg *cdim;
         int dind;
         for(dind=0;ret && dind<tab->ndimensions;dind++)
           {
            cdim=&tab->dimension[dind];
            skip_comments(fh);
            if(fscanf(fh,"%i",&cdim->var)==1)
              {
               if(cdim->var <= NUM_EQS)
                 {
                  skip_comments(fh);
                  if(fscanf(fh,"%li",&cdim->intervals)==1)
                    {
                     if(cdim->intervals <= MAXINTERVALS)
                       {
                        int iind;
                        for(iind=0;iind<cdim->intervals;iind++)
                          {
                           double x0,xf,sca,logb;
                           unsigned long nco;
                           int neqsel;
                           skip_comments(fh);
                           if(fscanf(fh,"%lf",&x0)==1 &&
                              fscanf(fh,"%lf",&xf)==1 &&
                              fscanf(fh,"%lu",&nco)==1 &&
                              fscanf(fh,"%lf",&sca)==1 &&
                              fscanf(fh,"%i",&neqsel)==1)
                             {
                              if(nco > 0)
                                {
                                 if(sca == 0.0 || (x0 > 0.0 && xf >= 0.0))
                                   {
                                    cdim->interval[iind].nco=nco;
                                    cdim->interval[iind].log=sca;
                                    cdim->interval[iind].neqsel=neqsel;
   //                                 printf("%g %g %lu %g\n",x0,xf,nco,sca);
                                    if(sca != 0.0)
                                      {
                                       logb=log(sca);
                                       cdim->interval[iind].c0=log(x0)/logb;
                                       cdim->interval[iind].cf=log(xf)/logb;
                                      }
                                    else
                                      {
                                       cdim->interval[iind].c0=x0;
                                       cdim->interval[iind].cf=xf;
                                      }
                                    if(nco > 1)
                                       cdim->interval[iind].ci=(cdim->interval[iind].cf - cdim->interval[iind].c0)/(nco-1);
                                    else
                                       cdim->interval[iind].ci=(cdim->interval[iind].cf - cdim->interval[iind].c0)+1.0;
                                   }
                                 else
                                   {
                                    lineerror(1,"*>The specified coordinates must be greater than 0 when using logarithmic scale");
                                    ret=0;
                                    break;
                                   }
                                }
                              else
                                {
                                 lineerror(1,"*>The number of coordinates must be at least 1");
                                 ret=0;
                                 break;
                                }
                             }
                           else
                             {
                              lineerror(1,"*>Can't read interval range specification");
                              ret=0;
                              break;
                             }
                          }
                       }
                     else
                       {
                        lineerror(1,"*>Too many intervals specified");
                        ret=0;
                        break;
                       }
                    }
                  else
                    {
                     lineerror(1,"*>Can't read number of intervals");
                     ret=0;
                     break;
                    }
                 }
               else
                 {
                  lineerror(1,"*>Incorrect number of equation specified");
                  ret=0;
                  break;
                 }
              }
             else
              {
               lineerror(1,"*>Reading number of equation");
               ret=0;
               break;
              }
           }
        }
      else
        {
         lineerror(1,"*>Incorrect number of table dimensions specified");
         ret=0;
        }
     }
   else
     {
      lineerror(1,"*>Reading number of table dimensions");
      ret=0;
     }
   return(ret);
  }

int load_conffile(char *cfgfile)
  {
   FILE *fh;
   int ret;
   ret=1;
   printf("Loading definition of tables from: %s...",cfgfile);
   fflush(stdout);
   fh=fopen(cfgfile,"rt");
   if(fh)
     {
      Currentline=1L;
      if(find_parameters(fh))
        {
         skip_comments(fh);
         if(fscanf(fh,"%i",&Confdef.nfiles)==1)
           {
            if(Confdef.nfiles <= MAXFILES)
              {
               int fi;
               for(fi=0;fi<Confdef.nfiles && ret;fi++)
                 {
                  skip_comments(fh);
                  if(fscanf(fh,"%"MAXIDSIZEC"s",Confdef.file[fi].ident)==1)
                    {
                     skip_comments(fh);
                     if(fscanf(fh,"%i",&Confdef.file[fi].ntables)==1)
                       {
                        skip_comments(fh);
                        if(Confdef.file[fi].ntables <= MAXTABLES)
                          {
                           int ci;
                           float *cons=(float *)&Confdef.file[fi].cons;
                           for(ci=0;ci<sizeof(struct consts)/sizeof(float);ci++)
                             {
                              skip_comments(fh);
                              if(fscanf(fh,"%f",cons+ci)!=1)
                                {
                                 lineerror(1,"*>Reading equation costants");
                                 ret=0;
                                 break;
                                }
                             }
                           if(ret)
                             {
                              int ti;
                              struct ttablecfg *tab;
                              for(ti=0;ti<Confdef.file[fi].ntables;ti++)
                                {
                                 int vi;
                                 tab=&Confdef.file[fi].table[ti];
                                 for(vi=0;vi<NUM_EQS;vi++)
                                   {
                                    skip_comments(fh);
                                    if(fscanf(fh,"%f",&tab->varinit[vi])!=1)
                                      {
                                       lineerror(1,"*>Reading variable initializations");
                                       ret=0;
                                       break;
                                      }
                                   }
                                 if(ret)
                                   {
                                    int nusedeqs;
                                    skip_comments(fh);
                                    if(fscanf(fh,"%i",&nusedeqs)==1)
                                      {
                                       if(nusedeqs <= NUM_EQS)
                                         {
                                          int ui,nsys,eq;
                                          for(nsys=0;nsys<NEQSYS;nsys++)
                                            {
                                             tab->ndiffeqs[nsys]=0;
                                             tab->nothereqs[nsys]=0;
                                            }
                                          for(ui=0;ui<nusedeqs;ui++)
                                            {
                                             skip_comments(fh);
                                             if(fscanf(fh,"%i",&eq)==1)
                                               {
                                                if(eq < NUM_EQS)
                                                  {
                                                   if(ui==0)
                                                      tab->outputvar=eq;
                                                   for(nsys=0;nsys<NEQSYS;nsys++)
                                                     {
                                                      if(Eq_sys[nsys][eq].diff)
                                                         tab->diffeqs[nsys][tab->ndiffeqs[nsys]++]=eq;
                                                      else
                                                         tab->othereqs[nsys][tab->nothereqs[nsys]++]=eq;
                                                     }
                                                  }
                                                else
                                                  {
                                                   lineerror(1,"*>Equation number out of range");
                                                   ret=0;
                                                   break;
                                                  }
                                               }
                                             else
                                               {
                                                lineerror(1,"*>Reading list of equation numbers");
                                                ret=0;
                                                break;
                                               }
                                            }
                                           if(ret)
                                            {
                                             ret=load_dimconf(fh, tab);
                                            }
                                         }
                                       else
                                         {
                                          lineerror(1,"*>Incorrect number of equations specified or inappropiate define");
                                          ret=0;
                                          break;
                                         }
                                      }
                                    else
                                      {
                                       lineerror(1,"*>Reading number of used equations in table");
                                       ret=0;
                                       break;
                                      }
                                   }
                                }
                             }
                          }
                        else
                          {
                           lineerror(1,"*>Incorrect number of tables in files specified or inappropiate define in source code");
                           ret=0;
                          }
                       }
                     else
                       {
                        lineerror(1,"*>Reading number of tables in files to be generated");
                        ret=0;
                       }
                    }
                  else
                    {
                     lineerror(1,"*>Reading file name string");
                     ret=0;
                    }
                 }
               if(ret)
                 {
                  char endid[3];
                  int nvar,nsys;
                  for(nvar=0;nvar<NUM_EQS;nvar++)
                    {
                     Varupdate[nvar]=0;
                     for(nsys=0;nsys<NEQSYS && !Varupdate[nvar];nsys++)
                        Varupdate[nvar]=Eq_sys[nsys][nvar].diff;
                    }
                  skip_comments(fh);
                  if(fscanf(fh,"%2s",endid) !=1 || strcmp("*/",endid))
                     lineerror(0,">End of file id not found, there may be data in the end of the file that will not be read");
                  puts("ok");
                 }
              }
            else
              {
               lineerror(1,"*>Incorrect number of table files specified or inappropiate define in source code");
               ret=0;
              }
           }
         else
           {
            lineerror(1,"*>Reading number of table files to be generated");
            ret=0;
           }
        }
      else
        {
         lineerror(1,"*>Cannot find start identification string");
         ret=0;
        }
      fclose(fh);
     }
   else
     {
      perror("*>Can't open the configuration file of tables");
      ret=0;
     }
   return(ret);
  }

struct tcoordinates get_coordinates_from_cfg(struct ttablecfg *tab)
  {
   struct tcoordinates coord;
   unsigned long i,j,k,ncoor,maxran,pos;
   struct tdimensioncfg *cdim;
   int ndims;

   ndims=tab->ndimensions;
   coord.size=(unsigned long *)malloc(ndims*sizeof(unsigned long));
   coord.posdim=(unsigned long *)malloc((ndims+1)*sizeof(unsigned long));
   if(coord.size && coord.posdim)
     {
      maxran=0;
      pos=1;
      for(i=0;i<ndims;i++)
        {
         cdim=&tab->dimension[i];
         coord.size[i]=0;
         for(j=0;j<cdim->intervals;j++)
            coord.size[i]+=cdim->interval[j].nco;
         maxran+=coord.size[i];
         coord.posdim[i]=pos;
         pos*=coord.size[i];
   //      printf("(siz:%lu pos:%lu)",coord.size[i],coord.posdim[i]);
        }
      coord.posdim[i]=pos;
   //   puts("");
      if(maxran > 0)
        {
         double basei,expon;
         coord.ranges=(float **)malloc(sizeof(float *)*ndims+sizeof(float)*maxran);
         coord.neqsel=(int **)malloc(sizeof(int *)*ndims+sizeof(int)*maxran);
         if(coord.ranges && coord.neqsel)
           {
            ncoor=0;
            for(i=0;i<ndims;i++) // create two bidimensional matrices
              {
               coord.ranges[i]=(float *)(coord.ranges+ndims)+ncoor;
               coord.neqsel[i]=(int *)(coord.neqsel+ndims)+ncoor;
               ncoor+=coord.size[i];
              }
            for(i=0;i<ndims;i++)
              {
               cdim=&tab->dimension[i];
               ncoor=0;
               for(j=0;j<cdim->intervals;j++)
                  for(k=0;k<cdim->interval[j].nco;k++)
                    {
                     coord.neqsel[i][ncoor]=cdim->interval[j].neqsel;
                     basei=cdim->interval[j].log;
                     if(k > cdim->interval[j].nco/2)
                        expon=cdim->interval[j].cf-cdim->interval[j].ci*(cdim->interval[j].nco-k-1);
                     else
                        expon=cdim->interval[j].c0+cdim->interval[j].ci*k;
                     if(basei==0.0)
                        coord.ranges[i][ncoor]=(float)expon;
                     else
                        coord.ranges[i][ncoor]=(float)pow(basei,expon);
   //                  printf("%g ",coord.ranges[i][ncoor]);
                     ncoor++;
                    }
   //            puts("");
              }
           }
         else
           {
            free(coord.neqsel);
            free(coord.ranges);
            free(coord.posdim);
            free(coord.size);
            perror("*>Allocating memory for table coordinates");
            coord.ranges=NULL;
           }
        }
      else
        {
         printf("*>The coordinates specified in the configuration file of tables has not the correct format\n");
         coord.ranges=NULL;
        }
     }
   else
     {
      free(coord.posdim);
      free(coord.size);
      perror("*>Allocating memory for table coordinate data");
      coord.ranges=NULL;
     }
   if(!coord.ranges) // if the allocation fails, set all pointers to NULL
     {
      coord.neqsel=NULL;
      coord.size=NULL;
      coord.posdim=NULL;
     }
   return(coord);
  }

int save_table(FILE *ofd, struct ttablecfg *tab, struct tcoordinates coord, float *table)
  {
   int ret;
   unsigned long tabsize,ndims;
   long i;
   ret=0;
   if(ofd)
     {
      printf("Saving...");
      fflush(stdout);
      ndims=tab->ndimensions;
      tabsize=coord.posdim[ndims];
      if(fwrite(&tabsize,sizeof(unsigned long),1,ofd) != 1)
         perror("Writing");
      else
         if(fwrite(&ndims,sizeof(unsigned long),1,ofd) != 1)
            perror("Writing");
         else
           {
            for(i=ndims-1;i>=0;i--)
              {
               if(fwrite(coord.size+i,sizeof(unsigned long),1,ofd) == 1)
                 {
                  if(!(fwrite(coord.ranges[i],sizeof(float),coord.size[i],ofd) == coord.size[i]))
                    {
                     perror("Writing file of tables");
                     break;
                    }
                 }
               else
                 {
                  perror("Writing");
                  break;
                 }
              }
            if(i<0)
              {
               if(fwrite(table,sizeof(float),tabsize,ofd) != tabsize)
                  perror("Writing");
               else
                 {
                  ret=1;
                  printf("Well done\n");
                 }
              }
           }
     }
   else
      perror("*>No Output file");
   return(ret);
  }

// Runge-Kutta approximation and function evaluation
inline float calculate_or_approximate_element(float h, struct ttablecfg *tab, struct consts *cons, union vars *var, int nsyseqs, float *outputvar)
  {
   int i,eq,num_diffeqs,num_othereqs;
   float k[4][NUM_EQS]; // intermediate numbers
   union vars varex1, varex2;
   int *diffeqorder,*othereqorder;

   num_diffeqs=tab->ndiffeqs[nsyseqs];
   diffeqorder=tab->diffeqs[nsyseqs];
   num_othereqs=tab->nothereqs[nsyseqs];
   othereqorder=tab->othereqs[nsyseqs];
   
   varex1=*var;
   do
     {
      varex1.named.t+=h/2;
      if(varex1.named.t != var->named.t) // check if the time-step (h) is affordable by a float
         break;
      h=h*2+FLT_MIN; // else increase it
     }
   while(1);

   varex2=*var;
   varex1=*var;
   for(i=0;i<num_othereqs;i++)
     {
      eq=othereqorder[i];
      varex2.list[eq+1]=Eq_sys[nsyseqs][eq].eq(&varex1,cons,0);
     }
   for(i=0;i<num_diffeqs;i++)
     {
      eq=diffeqorder[i];
      k[0][i]=h*Eq_sys[nsyseqs][eq].eq(&varex2,cons,0);
      varex1.list[eq+1]+=k[0][i]/2;
     }

   varex1.named.t+=h/2;
   varex2=*var;
   varex2.named.t+=h/2;
   for(i=0;i<num_othereqs;i++)
     {
      eq=othereqorder[i];
      varex1.list[eq+1]=Eq_sys[nsyseqs][eq].eq(&varex2,cons,h/2);
     }
   for(i=0;i<num_diffeqs;i++)
     {
      eq=diffeqorder[i];
      k[1][i]=h*Eq_sys[nsyseqs][eq].eq(&varex1,cons,h/2);
      varex2.list[eq+1]+=k[1][i]/2;
     }

   varex1=*var;
   varex1.named.t+=h/2;
   for(i=0;i<num_othereqs;i++)
     {
      eq=othereqorder[i];
      varex2.list[eq+1]=Eq_sys[nsyseqs][eq].eq(&varex1,cons,h/2);
     }
   for(i=0;i<num_diffeqs;i++)
     {
      eq=diffeqorder[i];
      k[2][i]=h*Eq_sys[nsyseqs][eq].eq(&varex2,cons,h/2);
      varex1.list[eq+1]+=k[2][i];
     }

   varex1.named.t+=h/2; // varex1.named.t=t+h
   varex2=varex1;
   for(i=0;i<num_othereqs;i++)
     {
      eq=othereqorder[i];
      varex1.list[eq+1]=Eq_sys[nsyseqs][eq].eq(&varex2,cons,h);
     }
   for(i=0;i<num_diffeqs;i++)
     {
      eq=diffeqorder[i];
      k[3][i]=h*Eq_sys[nsyseqs][eq].eq(&varex1,cons,h);
      var->list[eq+1]+=(k[0][i]+2*k[1][i]+2*k[2][i]+k[3][i])/6;
     }
   var->named.t=varex1.named.t;
   // only differential and other specified variables must be updated
   for(i=0;i<num_othereqs;i++) // update the other variables
     {
      eq=othereqorder[i];
      if(Varupdate[eq])
         var->list[eq+1]=varex1.list[eq+1];
     }
   *outputvar=(Eq_sys[nsyseqs][tab->outputvar].diff)?var->list[tab->outputvar+1]:varex1.list[tab->outputvar+1];
   Function_evaluations++;
   
   return(h);
  }

// Runge-Kutta approximation only
inline float approximate_element(float h, struct ttablecfg *tab, struct consts *cons, union vars *var, int nsyseqs, float *outputvar)
  {
   int i,eq,num_diffeqs;
   float k[4][NUM_EQS]; // intermediate numbers
   union vars varex1, varex2;
   int *diffeqorder;

   num_diffeqs=tab->ndiffeqs[nsyseqs];
   diffeqorder=tab->diffeqs[nsyseqs];
   
   varex1=*var;
   do
     {
      varex1.named.t+=h/2;
      if(varex1.named.t != var->named.t) // check if the time-step (h) is affordable by a float
         break;
      h=h*2+FLT_MIN; // else increase it
     }
   while(1);

   varex2=*var;
   varex1=*var;
   for(i=0;i<num_diffeqs;i++)
     {
      eq=diffeqorder[i];
      k[0][i]=h*Eq_sys[nsyseqs][eq].eq(&varex2,cons,0);
      varex1.list[eq+1]+=k[0][i]/2;
     }

   varex1.named.t+=h/2;
   varex2=*var;
   for(i=0;i<num_diffeqs;i++)
     {
      eq=diffeqorder[i];
      k[1][i]=h*Eq_sys[nsyseqs][eq].eq(&varex1,cons,h/2);
      varex2.list[eq+1]+=k[1][i]/2;
     }

   varex2.named.t+=h/2;
   varex1=*var;
   for(i=0;i<num_diffeqs;i++)
     {
      eq=diffeqorder[i];
      k[2][i]=h*Eq_sys[nsyseqs][eq].eq(&varex2,cons,h/2);
      varex1.list[eq+1]+=k[2][i];
     }

   varex1.named.t+=h;
   for(i=0;i<num_diffeqs;i++)
     {
      eq=diffeqorder[i];
      k[3][i]=h*Eq_sys[nsyseqs][eq].eq(&varex1,cons,h);
      var->list[eq+1]+=(k[0][i]+2*k[1][i]+2*k[2][i]+k[3][i])/6;
     }
   var->named.t=varex1.named.t;

   *outputvar=(Eq_sys[nsyseqs][tab->outputvar].diff)?var->list[tab->outputvar+1]:varex1.list[tab->outputvar+1];
   
   Function_evaluations++;
   
   return(h);
  }
inline int divergent_value(float num)
  {
   return(num != num || num > FLT_MAX || num < -FLT_MAX);
  }

inline float check_number(float num, float alt_num)
  {
   float ret;
   if(divergent_value(num))
     {
      ret=alt_num;
      Numeric_errs++;
     }
   else
     ret=num;
   return(ret);
  }

// Richardson extrapolation
inline float step_reduction_approximation(float h, struct ttablecfg *tab, struct consts *cons, union vars *var, int nsyseqs, float *outputvar)
  {
   int i,eq,num_diffeqs,num_othereqs;
   float actual_h, current_h, best_h; // step size
   union vars var2,var1,var21,var21b;
   float var21err, var21berr, varcerr;
   int overerr,cdepth,var21bactu;
   float outv1,outv2;
   int *diffeqorder,*othereqorder;
   float (*calculate_element)(float, struct ttablecfg *, struct consts *, union vars *, int, float *);
   
   num_diffeqs=tab->ndiffeqs[nsyseqs];
   diffeqorder=tab->diffeqs[nsyseqs];
   num_othereqs=tab->nothereqs[nsyseqs];
   othereqorder=tab->othereqs[nsyseqs];
   // just to speed up calculation
   calculate_element=(num_othereqs == 0)?approximate_element:calculate_or_approximate_element;
   best_h=current_h=h;
   var21berr=FLT_MAX;
   cdepth=Max_retries;
   do
     {
      var2=*var;
      var1=*var;
      var21b=*var;
      actual_h=calculate_element(current_h/2, tab, cons, &var2, nsyseqs, &outv2);
      actual_h+=calculate_element(current_h/2, tab, cons, &var2, nsyseqs, &outv2);
      calculate_element(actual_h, tab, cons, &var1, nsyseqs, &outv1);
      *outputvar=(16*outv2-outv1)/15;
      overerr=0;
      var21err=0;
      var21=var2; // to preserve value of untouched variables
      for(i=0;i<num_diffeqs;i++)
        {
         eq=diffeqorder[i];
         var21.list[eq+1]=(16*var2.list[eq+1]-var1.list[eq+1])/15;
         varcerr=fabs(var2.list[eq+1]-var1.list[eq+1]);
         var21err+=varcerr;
         overerr=overerr || divergent_value(varcerr) || (varcerr > fabs(Error_tolerance*var21.list[eq+1]));
        }
      if(var21berr>var21err) // also false if var21err is NaN
        {
         var21berr=var21err;
         var21b=var21;
         best_h=actual_h;
         cdepth=Max_retries;
         var21bactu=1;
//         printf("A");
        }
      else
        {
         cdepth--;
         var21bactu=0;
//         printf("B");
        }
      if(overerr && current_h > FLT_MIN && (var21bactu || cdepth)) // not enough precision, retry
        {
//         printf("C%1.5f[%1.6f](%i) ",var21err,current_h,cdepth);
         if(var21err != var21err || var21err > 16/Error_tolerance) // 16 is arbitrary, not quite important
            var21err=16/Error_tolerance;
         current_h=current_h/(Time_step_folding+log10(1+var21err/(Error_tolerance/16)));

//         printf("%g ",current_h);d
        }
      else
        {
//         printf("F%g[%g](%i) ",var21err,current_h,cdepth);
         break;
        }
     }
   while(1);
   for(i=0;i<num_diffeqs;i++)
     {
      eq=diffeqorder[i];
//          if(var21.list[eq+1] == var21.list[eq+1])
      var->list[eq+1]=check_number(var21b.list[eq+1], tab->varinit[eq]);
     }
//         *var=var21;
   var->named.t=var2.named.t;
   for(i=0;i<num_othereqs;i++)
     {
      eq=othereqorder[i];
//            if(var2.list[1+eq] == var2.list[1+eq])
      var->list[eq+1]=check_number(var21b.list[eq+1], tab->varinit[eq]);
     }
//         if(*outputvar != *outputvar)
//            *outputvar=tab->varinit[tab->outputvar];
   *outputvar=check_number(*outputvar, tab->varinit[tab->outputvar]);
   return(best_h);
  }

inline float adaptive_step_approximation(float h, struct ttablecfg *tab, struct consts *cons, union vars *var, float *time_step, int nsyseqs, float *outputvar)
  {
   float leftover_h,cutime_step,actime_step;
   var->named.t-=h; // kunge-kutta method needs t to calculate Yt+h
   leftover_h=h;
//   printf("{%f ",*time_step);
   do
     {
      cutime_step=(*time_step*1.5 < leftover_h)?*time_step:leftover_h;
//      if(*time_step < leftover_h) printf("1"); else printf("2");
      actime_step=step_reduction_approximation(cutime_step, tab, cons, var, nsyseqs, outputvar);
      leftover_h-=actime_step;
      if(actime_step<3*cutime_step/4 && actime_step > FLT_MIN) // If the used time step has changed and is not 0
         *time_step=actime_step;
      else
        {
         if(cutime_step==*time_step)
            *time_step*=Time_step_folding;
        }
     }
   while(leftover_h > FLT_MIN);
//   printf("}");
   return(h-leftover_h);
  }

inline void function_element_calculation(float h, struct ttablecfg *tab, struct consts *cons, union vars *var, int nsyseqs, float *outputvar)
  {
   int i,eq,num_othereqs;
   union vars varex1;
   int *othereqorder;

   num_othereqs=tab->nothereqs[nsyseqs];
   othereqorder=tab->othereqs[nsyseqs];
   
   for(i=0;i<num_othereqs;i++)
     {
      eq=othereqorder[i];
      varex1.list[eq+1]=Eq_sys[nsyseqs][eq].eq(var,cons,h);
      if(Varupdate[eq])
         var->list[eq+1]=varex1.list[eq+1];
     }
   *outputvar=(Eq_sys[nsyseqs][tab->outputvar].diff)?var->list[tab->outputvar+1]:varex1.list[tab->outputvar+1];
   
   Function_evaluations++;
  }

inline float table_element_calculation(float h, struct ttablecfg *tab, struct consts *cons, union vars *var, float *time_step, int nsyseqs, float *outputvar)
  {
   float ret_h;
   if(tab->ndiffeqs[nsyseqs] == 0)
     {
      function_element_calculation(h, tab, cons, var, nsyseqs, outputvar);
      ret_h=h;
     }
   else
      ret_h=adaptive_step_approximation(h, tab, cons, var, time_step, nsyseqs, outputvar);
   return(ret_h);
  }

inline int inc_coordinates(unsigned long *coo, int *dper, int *vper, struct ttablecfg *tab, struct tcoordinates *coord, union vars *vars, float *prev_t)
  {
   unsigned long ndims, idim, ieq;
   int ret;
   ndims=tab->ndimensions;
   
   for(idim=0;idim<ndims && ++coo[idim] >= coord->size[dper[idim]];idim++)
     {
      coo[idim]=0;
     }
   if(idim<ndims) // if calculation has not finished yet
     {
      if(idim>0 || vper[0] != 0) // if a dimension coordinate which is not time has changed or the table does not have time dimension
        {
         // initialize all variables except time (which does not have initialization value)
         for(ieq=0;ieq<NUM_EQS;ieq++)
            vars->list[ieq+1]=tab->varinit[ieq];
         // update all variables except the first one (normally time)
         for(idim=1;idim<ndims;idim++)
            vars->list[vper[idim]]=coord->ranges[dper[idim]][coo[idim]];
         *prev_t=(vper[0] == 0)?coord->ranges[dper[0]][0]:0; // first time coordinate or 0
        }
      vars->list[vper[0]]=coord->ranges[dper[0]][coo[0]]; // update first-dimension variable (normally time)
      ret=1;
     }
   else
      ret=0;
   return(ret);
  }

void calculate_table(float *table, struct ttablecfg *tab, struct consts *cons, struct tcoordinates coord)
  {
   unsigned long coo[MAXDIMENSIONS]={0}; // all elements are initialized to 0
   float prev_t,next_t,time_step;
   unsigned long tabpos,ndims,i;
   union vars vars;
   int dimper[MAXDIMENSIONS], dimvarper[MAXDIMENSIONS];
   int neqsel,neqsys;

   ndims=tab->ndimensions;
   // time must be the first dimension
   // find and place it in the first position
   for(i=0;i<ndims;i++)
     {
      dimper[i]=i;
      if(tab->dimension[i].var==0)
        {
         dimper[i]=0;
         dimper[0]=i;
        }
     }
   // calculate the variable that will be incremented by each dimension
   for(i=0;i<ndims;i++)
      dimvarper[i]=tab->dimension[dimper[i]].var;

   // initialize all variables except t
   for(i=0;i<NUM_EQS;i++)
       vars.list[i+1]=tab->varinit[i];
   // initialize coordinate-dependent variables
   for(i=0;i<ndims;i++)
      vars.list[tab->dimension[i].var]=coord.ranges[i][coo[i]]; // coo[] is initialized to 0
   if(dimvarper[0] != 0) // if the first coordinate-dependent variable is not time
      prev_t=0; // the tables does not have time dimension: set time to 0
   else
      prev_t=coord.ranges[dimper[0]][coo[0]]; // first initial-dimension (normally time) coordinate
   time_step=1; // Not important
   do
     {
      tabpos=0;
      neqsel=0;
      for(i=0;i<ndims;i++)
        {
         tabpos+=coo[dimper[i]]*coord.posdim[i]; // calculate table position
         neqsel|=coord.neqsel[i][coo[i]]; // calculate number of equation selector to be used for this element
        }
      neqsys=Eq_sel[neqsel](&vars, cons);
      next_t=(dimvarper[0] == 0)?coord.ranges[dimper[0]][coo[0]]:0.0; // if table has time dimension, calculate current time coordinate
      table_element_calculation(next_t-prev_t, tab, cons, &vars, &time_step, neqsys, &table[tabpos]);
      prev_t=next_t;
/*      if(Function_evaluations > 10000)
        {
         Function_evaluations=0;
         for(i=0;i<ndims;i++)
            printf("%f ",coord.ranges[i][coo[dimper[i]]]);
         puts("");
        }*/
     }
   while(inc_coordinates(coo, dimper, dimvarper, tab, &coord, &vars, &prev_t));
  }

struct ttable generate_table(FILE *ofd, struct ttablecfg *tabcfg, struct consts *cons)
  {
   unsigned long tabsize;
   struct ttable table;
   table.elems=NULL;
   
   table.coord=get_coordinates_from_cfg(tabcfg);
   if(table.coord.ranges)
     {
      printf("Calculating...");
      fflush(stdout);
      tabsize=table.coord.posdim[tabcfg->ndimensions];
      table.elems=(float *)malloc(sizeof(float)*tabsize);
      if(table.elems)
        {
         clock_t startt,endt;
         Function_evaluations=0L;
         Numeric_errs=0L;
         startt=clock();
         calculate_table(table.elems, tabcfg, cons, table.coord);
         endt=clock();
         printf("%lu evaluations (%gs) Numeric errors: %lu\n",Function_evaluations,(endt-startt)/(float)CLOCKS_PER_SEC,Numeric_errs);
         save_table(ofd, tabcfg, table.coord, table.elems);
        }
      else
        {
         free(table.coord.ranges);
         table.coord.ranges=NULL;
         perror("*>Table file size too large (Not enough memory)");
        }
     }
   else
     {
      perror("*>Too many coordinates (Not enough memory)");
     }
   return(table);
  }

int generate_files(void)
  {
   FILE *ofd;
   int ret;
   long ti;
   struct ttablecfg *tab;
   
   ret=1;
   for(Currentfile=0;Currentfile<Confdef.nfiles;Currentfile++)
     {
      ofd=fopen(Confdef.file[Currentfile].ident,"w");
      if(ofd)
        {
         struct consts *cons;
         cons=&Confdef.file[Currentfile].cons;
         printf("--File: %s\n",Confdef.file[Currentfile].ident);
         for(ti=0;ti<Confdef.file[Currentfile].ntables;ti++)
           {
            tab=&Confdef.file[Currentfile].table[ti];
            Tables[ti]=generate_table(ofd, tab, cons);
            if(!Tables[ti].elems)
              {
               printf("*>Table could not be processed\n");
               ret=0;
              }
           }
         for(ti--;ti>=0;ti--)
           {
            free(Tables[ti].elems);
            free(Tables[ti].coord.neqsel);
            free(Tables[ti].coord.ranges);
            free(Tables[ti].coord.posdim);
            free(Tables[ti].coord.size);
           }
         fclose(ofd);
        }
      else
        {
         perror("*>Can't create table file");
         ret=0;
        }
     }
   return(ret);
  }

int main()
  {
   int ret;
   ret=load_conffile("tab2cfg.c");
   if(ret)
      generate_files();
   return(!ret);
  }
