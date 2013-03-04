/*!
\file  
\brief This file contains function prototypes

\date Started 1/18/07
\author George
\version\verbatim $Id: proto.h 9628 2011-03-23 21:15:43Z karypis $ \endverbatim
*/

#ifndef _SIMPAT_PROTO_H_
#define _SIMPAT_PROTO_H_

#ifdef __cplusplus
extern "C"
{
#endif

/* cmdline.c */
void cmdline_parse(params_t *ctrl, int argc, char *argv[]);


/* main.c */
void ComputeNeighbors(params_t *params);


#ifdef __cplusplus
}
#endif

#endif 
