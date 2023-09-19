#pragma once

#define Luminosita(x) x##.0

#define NO_GUI   0
#define GRAPHICS 1


#define GET_VIDEO_FRAMERATE "ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate ./video/test.mp4"
#define GET_VIDEO_DURATION  "ffprobe -i ./video/test.mp4 -v quiet -show_entries format=duration -hide_banner -of default=noprint_wrappers=1:nokey=1"
#define GET_VIDEO_FRAME     "ffmpeg  -i ./video/test.mp4 select='between(n,%d,%d)' -frames:v 1 ./frames/%03d.bmp"
#define GET_VIDEO_WIDTH     "ffprobe -v error -select_streams v:0 -show_entries stream=width -of default=nw=1:nk=1 ./video/test.mp4"
#define GET_VIDEO_HEIGHT    "ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=nw=1:nk=1 ./video/test.mp4"

