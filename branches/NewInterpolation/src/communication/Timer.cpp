/***************************************************************************
 *                           Timer.cpp                 					   *
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

#include "../../include/communication/Timer.h"

//( Timer

Timer::Timer()
{
  init();
}

Timer::~Timer()
{
}

void Timer::init()
{
  gettimeofday(&t_start,NULL);
  t_pause_cumul.tv_sec = 0;
  t_pause_cumul.tv_usec = 0;
  paused_time=0;
}

void Timer::pause()
{
  gettimeofday(&t_pause_start,NULL);
  paused_time=getTimeS();
  paused = true;
}

void Timer::resume()
{
  gettimeofday(&t_pause_end,NULL);
  t_pause_cumul.tv_sec += t_pause_end.tv_sec - t_pause_start.tv_sec;
  t_pause_cumul.tv_usec += t_pause_end.tv_usec - t_pause_start.tv_usec;
  paused = false;
}

double Timer::getTimeS()
{
  if(!paused)
    {
      gettimeofday(&t_end,NULL);
      return ((t_end.tv_sec-t_start.tv_sec-t_pause_cumul.tv_sec) + (t_end.tv_usec-t_start.tv_usec-t_pause_cumul.tv_usec)*1e-6);
    }
  else
    return paused_time;
}

double Timer::getTimeMs()
{
  if(!paused)
    {
      gettimeofday(&t_end,NULL);
      return ((t_end.tv_sec-t_start.tv_sec-t_pause_cumul.tv_sec)*1000. + (t_end.tv_usec-t_start.tv_usec-t_pause_cumul.tv_usec)*1e-3);
    }
  else
    return paused_time*1000.;
}


//) Timer


//) timer.cpp
