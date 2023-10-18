hellomake: main.c
     mpic++ -o a -I/opt/homebrew/Cellar/opencv/4.8.1/include/opencv4 -I/opt/homebrew/Cellar/sdl2/2.28.3/include/SDL2 \
     -std=c++11 \
     -lSDL2 -lSDL2_ttf -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio \
