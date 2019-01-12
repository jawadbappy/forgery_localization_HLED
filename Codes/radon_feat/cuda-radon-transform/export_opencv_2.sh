#!/bin/bash

THISSCRIPTPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OCVBASEPATH=$THISSCRIPTPATH/opencv-2.4.13/build_release
echo $OCVBASEPATH


if ! [[ :$LIBRARY_PATH: == *":$OCVBASEPATH/lib:"* ]] ; then
  export LIBRARY_PATH=$OCVBASEPATH/lib:$LIBRARY_PATH
fi
if ! [[ :$LD_LIBRARY_PATH: == *":$OCVBASEPATH/lib:"* ]] ; then
  export LD_LIBRARY_PATH=$OCVBASEPATH/lib:$LD_LIBRARY_PATH
fi

if ! [[ :$CPATH: == *":$OCVBASEPATH/include:"* ]] ; then
  export CPATH=$OCVBASEPATH/include:$CPATH
fi
if ! [[ :$C_INCLUDE_PATH: == *":$OCVBASEPATH/include:"* ]] ; then
  export C_INCLUDE_PATH=$OCVBASEPATH/include:$C_INCLUDE_PATH
fi
if ! [[ :$CPLUS_INCLUDE_PATH: == *":$OCVBASEPATH/include:"* ]] ; then
  export CPLUS_INCLUDE_PATH=$OCVBASEPATH/include:$CPLUS_INCLUDE_PATH
fi

if ! [[ :$CMAKE_PREFIX_PATH: == *":$OCVBASEPATH:"* ]] ; then
  export CMAKE_PREFIX_PATH=$OCVBASEPATH:$CMAKE_PREFIX_PATH
fi

