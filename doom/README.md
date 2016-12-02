# doom

Linux installation issues:

Would not run on remote server with x11, so followed: https://github.com/Marqt/ViZDoom/issues/36

Before `game.init()` added `game.add_game_args("+vid_forcesurface 1")`

solved the problem and runs



## OS X installation notes:

`brew install python3`, also use pip for python3 as here:

http://stackoverflow.com/questions/20082935/how-to-install-pip-for-python3-on-mac-os-x


In CMakeLists.txt comment out lines 222-235: https://github.com/Marqt/ViZDoom/blob/master/CMakeLists.txt#L222
to make it compile on OS X 10.12 (Sierra)


Then:


```
brew install boost

brew install boost-python --with-python3 --without-python

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON3=ON -DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python3/3.5.2_3/Frameworks/Python.framework/Versions/3.5/include/python3.5m -DPYTHON_LIBRARY=/usr/local/Cellar/python3/3.5.2_3/Frameworks/Python.framework/Versions/3.5/lib/libpython3.5m.dylib

make

rm bin/vizdoom

ln -s vizdoom.app/Contents/MacOS/vizdoom bin/vizdoom

cd examples/python

python basic.py
```

Look at issues:

https://github.com/Marqt/ViZDoom/issues/147

https://github.com/Marqt/ViZDoom/issues/144

etc.


## OS X SDL issue:

I was getting this error on my OS X machine:

```
Scanning dependencies of target vizdoom
[ 35%] Building CXX object src/vizdoom/src/CMakeFiles/vizdoom.dir/__autostart.cpp.o
[ 36%] Building C object src/vizdoom/src/CMakeFiles/vizdoom.dir/posix/sdl/crashcatcher.c.o
[ 36%] Building CXX object src/vizdoom/src/CMakeFiles/vizdoom.dir/posix/sdl/hardware.cpp.o
/Users/eugenioculurciello/Desktop/ViZDoom-master/src/vizdoom/src/posix/sdl/hardware.cpp:83:37: error: use of undeclared identifier
      'SDL_GetCurrentVideoDriver'
                Printf("Using video driver %s\n", SDL_GetCurrentVideoDriver());
                                                  ^
1 error generated.
make[2]: *** [src/vizdoom/src/CMakeFiles/vizdoom.dir/posix/sdl/hardware.cpp.o] Error 1
make[1]: *** [src/vizdoom/src/CMakeFiles/vizdoom.dir/all] Error 2
make: *** [all] Error 2```

I went into: ViZDoom-master/src/vizdoom/src/posix/sdl/
and changed all references:

`#include <SDL.h> to #include <SDL2/SDL.h>`
