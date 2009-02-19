/***************************************************************************
 *                           Timer.h                                       *
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

#ifndef TIMER_H
#define TIMER_H

#include <iostream>

#include <sys/time.h>

//( Timer

/**
 * record the time of the simulation
 *
 * @author 
 **/
class Timer 
{
public:
  /**
   * Default constructor
   * 
   **/
  Timer();

  /**
   * Default destructor
   *
   **/
  ~Timer();

  /**
   *
   * set start value to current time
   * 
   *
   * @return  void
   *
   **/
  void init();

  /**
   *
   * pause the timer
   * 
   *
   * @return  void
   *
   **/
  void pause();

  /**
   *
   * after a pause, relaunch the timer
   * 
   *
   * @return  void
   *
   **/
  void resume();

  /**
   *
   *
   * @return  time elapsed since last timer init in s
   *
   **/
  double getTimeS();

  /**
   *
   *
   * @return  time elapsed since last timer init in ms
   *
   **/
  double getTimeMs();

protected:
  /**
   *
   * start time
   * 
   **/
  struct timeval t_start;

  /**
   *
   * end time
   * 
   **/
  struct timeval t_end;

  /**
   *
   * accumulate pause time
   * 
   **/
  struct timeval t_pause_cumul;

  /**
   *
   * start pause time
   * 
   **/
  struct timeval t_pause_start;

  /**
   *
   * end pause time
   * 
   **/
  struct timeval t_pause_end;

  /**
   *
   * true if the timer is paused
   * 
   **/
  bool paused;

  /**
   *
   * display the time as it was when paused (in s)
   * 
   **/
  double paused_time;
}
;

//) Timer


#endif

//) timer.hpp
