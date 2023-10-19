#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "/opt/homebrew/Cellar/open-mpi/4.1.5/include/mpi.h"

#include "/opt/homebrew/Cellar/sdl2/2.28.3/include/SDL2/SDL.h"
#include "/opt/homebrew/Cellar/sdl2_image/2.6.3_2/include/SDL2/SDL_image.h"
#include "/opt/homebrew/Cellar/sdl2_ttf/2.20.2/include/SDL2/SDL_ttf.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "utility.h"

int PIXEL_SCALE = 8;
int operation_mode = 1;
char video_path[256] {0};
 
int width  = 0, 
    height = 0;

#define ASCII_WIDTH width
#define ASCII_HEIGHT height

#define FONT_SIZE 4

int nFrames,framerate = 0;

MPI_Datatype sdl_color;

cv::VideoCapture videoStream;
int videoStreamIndex = -1;

int initOpenCV() {
   videoStream = cv::VideoCapture(video_path);

    if (!videoStream.isOpened()) {
        std::cout << "Cannot open the video file.\n";
        return -1;
    }

    return 0;
}


// Array di caratteri ASCII ordinati in base alla "luminosità"
// Modifica o aggiungi caratteri se desideri un diverso effetto ASCII art
const char asciiChars[] = " .:-=+*#%@";

const int numChars = sizeof(asciiChars) - 1; // Numero di caratteri, escludendo il terminatore di stringa '\0'

// Funzione per convertire il colore in scala di grigi
inline uint8_t grayscale(uint8_t r, uint8_t g, uint8_t b) {
    return (r+g+b) / 3;
}

inline uint8_t getCharIndex(uint8_t grayscale){
    return grayscale * numChars / 256;
}

// Funzione per ottenere il carattere ASCII corrispondente a un determinato valore di scala di grigi
inline char getAsciiChar(uint8_t grayscale) {
    // Mappatura del valore di scala di grigi all'indice del carattere ASCII
    uint8_t index = getCharIndex(grayscale);
    return asciiChars[index];
}

void initializeSDL(SDL_Window **window, SDL_Renderer **renderer, TTF_Font **font) {
    SDL_Init(SDL_INIT_VIDEO);
    if (operation_mode == GRAPHICS)
    {
        *window = SDL_CreateWindow("Ascii video", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, ASCII_WIDTH * PIXEL_SCALE, ASCII_HEIGHT * PIXEL_SCALE, 0);
        if (!(*window)) {
            printf("Errore durante la creazione della finestra: %s\n", SDL_GetError());
            MPI_Finalize();
            exit(1);
        }
        
        *renderer = SDL_CreateRenderer(*window, -1, 0);
        if (!(*renderer)) {
            printf("Errore durante la creazione del renderer: %s\n", SDL_GetError());
            SDL_DestroyWindow(*window);
            MPI_Finalize();
            exit(1);
        }
    }
    if (TTF_Init() == -1) {
        printf("Errore durante l'inizializzazione della libreria TTF: %s\n", TTF_GetError());
        SDL_DestroyRenderer(*renderer);
        SDL_DestroyWindow(*window);
        MPI_Finalize();
        exit(1);
    }
    *font = TTF_OpenFont("sans.ttf", FONT_SIZE);
    if (!(*font)) {
        printf("Errore durante il caricamento del font: %s\n", TTF_GetError());
        TTF_Quit();
        SDL_DestroyRenderer(*renderer);
        SDL_DestroyWindow(*window);
        MPI_Finalize();
        exit(1);
    }
}

void destroySDL(SDL_Window *window, SDL_Renderer *renderer, TTF_Font *font) {
    TTF_CloseFont(font);
    TTF_Quit();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

SDL_Window *window = NULL;
SDL_Renderer *renderer = NULL;
TTF_Font *font = NULL;

unsigned char *imagePixels = NULL;

unsigned char *allAsciiArtIdx = NULL;
SDL_Color *allAsciiArtPixelColor = NULL; 
SDL_Texture **asciiTextures = NULL;
SDL_Color *asciiArtPixelColor = NULL;

MPI_Comm comm2D;

int rank_first, rank_last;
int rank_up, rank_down;
int localHeight;
int localWidth;

// Create 2D topology
int dims[2] = {0};
int periods[2] = {0};
int coords[2] = {0};

void init(int rank, int size){

    MPI_Type_contiguous(sizeof(SDL_Color) , MPI_BYTE , &sdl_color);
    MPI_Type_commit(&sdl_color);

   
    MPI_Dims_create(size, 1, dims);
    
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 1, &comm2D);
    MPI_Comm_rank(comm2D, &rank);
    MPI_Comm_size(comm2D, &size);
    MPI_Cart_coords(comm2D, rank, 2, coords);

    
    MPI_Cart_shift(comm2D, 0, 1, &rank_up, &rank_down);

    const int top_coords[] = {0};
    const int bottom_coords[] = {size-1};

    MPI_Cart_rank(comm2D , top_coords , &rank_first);
    MPI_Cart_rank(comm2D , bottom_coords , &rank_last);

    if (rank == rank_first) {
        if (initOpenCV() != 0) {
            printf("Failed to initialize OpenCV\n");
            return;
        }

        framerate = videoStream.get(cv::CAP_PROP_FPS);
        width = videoStream.get(cv::CAP_PROP_FRAME_WIDTH);
        height = videoStream.get(cv::CAP_PROP_FRAME_HEIGHT);
        nFrames = videoStream.get(cv::CAP_PROP_FRAME_COUNT);

        allAsciiArtPixelColor = (SDL_Color *)malloc((ASCII_WIDTH * ASCII_HEIGHT + 1) * sizeof(SDL_Color));
        allAsciiArtIdx = (unsigned char *)malloc((ASCII_WIDTH * ASCII_HEIGHT + 1) * sizeof(unsigned char) * 3);
        asciiTextures = (SDL_Texture**)malloc(numChars * sizeof(SDL_Texture*));

        memset(asciiTextures, 0, numChars * sizeof(SDL_Texture*));
        memset(allAsciiArtIdx, 0, (ASCII_WIDTH * ASCII_HEIGHT + 1) * sizeof(unsigned char) * 3);
        memset(allAsciiArtPixelColor, 0, (ASCII_WIDTH * ASCII_HEIGHT + 1) * sizeof(SDL_Color));

        printf("%d, %d\n", width, height);

        if (operation_mode == GRAPHICS)
            initializeSDL(&window, &renderer, &font);

        SDL_Color textColor = {255, 255, 255, 255};
        for (int i = 0; i < numChars; ++i){
            SDL_Surface *asciiSurface = TTF_RenderText_Solid(font, &asciiChars[i], textColor);
            asciiTextures[i] = SDL_CreateTextureFromSurface(renderer, asciiSurface);
        }

        asciiArtPixelColor = allAsciiArtPixelColor;
    }

    MPI_Bcast(&width, 1, MPI_INT, rank_first, comm2D);
    MPI_Bcast(&height, 1, MPI_INT, rank_first, comm2D);
    MPI_Bcast(&nFrames, 1, MPI_INT, rank_first, comm2D);
    MPI_Bcast(&framerate, 1, MPI_INT, rank_first, comm2D);

    localHeight = ASCII_HEIGHT / dims[0];
    localWidth = ASCII_WIDTH;

    if (rank != rank_first){
        asciiArtPixelColor = (SDL_Color *)malloc((localWidth * localHeight + 1) * sizeof(SDL_Color));
        memset(asciiArtPixelColor, 0, (localWidth * localHeight + 1) * sizeof(SDL_Color));
    }
}

void processFrames(int rank, int size) { 

    int startX = 0;
    int startY = coords[0] * localHeight *localWidth;

    unsigned char *asciiArtIdx = 0;

    int quit = 0;
    cv::Mat frame;
    SDL_Event event;
    MPI_Request request;

    int thisRankFrameSize;
    int thisRankFinalFrameSize = 0;

    if (rank == rank_first){ 
        asciiArtIdx = allAsciiArtIdx;

        videoStream.set(cv::CAP_PROP_POS_FRAMES, 0);

        if (!videoStream.read(frame)) {
            printf("Failed to extract frame\n");
            quit = 1;
            MPI_Bcast(&quit, 1, MPI_INT, rank_first, comm2D);
            return;
        }

        // Calcola la dimensione corretta della porzione di dati da inviare
        int sendSize = (localHeight * localWidth * 3);

        // Invia solo la porzione di dati richiesta a destra
        for (int i = 0; i < size; i++)
            if (i != rank_first)
                MPI_Isend(&frame.data[localHeight * localWidth * 3 * i], sendSize, MPI_CHAR, i, 0, comm2D, &request);
    }
    
    for (int i = 0, framesdecoded = 0; i < nFrames && quit == 0; i ++, framesdecoded += size){
        #pragma region Chiudi_Programma
            if (operation_mode == GRAPHICS)
            {
                if (rank == rank_first) {
                    SDL_PollEvent(&event);
                    switch (event.type) {
                        case SDL_QUIT:
                            quit = 1;
                            break;
                    }
                }
                MPI_Bcast(&quit, 1, MPI_INT, rank_first, comm2D);
            }
        #pragma endregion
        
        //L'immagine sarà trasmessa da sinistra verso destra (quindi non per forza il rank 0 sarà il primo)

            if (rank != rank_first) {
                #pragma region Ricevi_frame_e_invia_porzione
                    MPI_Status status;

                    // Ottieni la dimensione del messaggio in arrivo
                    // Alloca la memoria necessaria per ricevere il messaggio
                    if (imagePixels == NULL) {
                        MPI_Probe(rank_first, 0, comm2D, &status);
                        MPI_Get_count(&status, MPI_CHAR, &thisRankFrameSize);
                        imagePixels = (uint8_t*)malloc(thisRankFrameSize * sizeof(uint8_t));
                    }

                    // Ricevi il buffer dei pixel
                    MPI_Recv(imagePixels, thisRankFrameSize, MPI_CHAR, rank_first, 0, comm2D, &status);
                #pragma endregion
            }
        
        #pragma region Decodifica_frame

            unsigned char* pixels = (imagePixels == NULL) ? &frame.data[0] : &imagePixels[0];
            unsigned char* base = (imagePixels == NULL) ? &frame.data[0] : &imagePixels[0];
            for (int y = 0; y < localHeight; y++) {
                for (int x = 0; x < localWidth; x++) {

                    uint8_t b = *pixels++;
                    uint8_t g = *pixels++;
                    uint8_t r = *pixels++;

                    int mid_b = 0, mid_r = 0, mid_g = 0, num_pixels = 0, luminance = 0;

                    for (int j = -4; j <= 4; j++) {
                        for (int i = -4; i <= 4; i++) {
                            int ny = y + j;
                            int nx = x + i;

                            if (ny >= 0 && ny < localHeight && nx >= 0 && nx < localWidth) {
                                luminance += 0.2989 * base[ny * localWidth + nx + 0] + 0.5870 * base[ny * localWidth + nx + 1] + 0.1140 * base[ny * localWidth + nx + 2];
                                num_pixels++;
                            }
                        }
                    }

                    if (num_pixels) luminance/=num_pixels*1.1f;

                    asciiArtPixelColor[y * localWidth + x] = SDL_Color{b,g,r, getCharIndex(luminance)};
                }
            }

        #pragma endregion
       
        
        if (rank == rank_first) {
            videoStream.set(cv::CAP_PROP_POS_FRAMES, i+1);

            if (!videoStream.read(frame)) {
                printf("Failed to extract frame\n");
                quit = 1;
                MPI_Bcast(&quit, 1, MPI_INT, rank_first, comm2D);
                break;
            }

            // Calcola la dimensione corretta della porzione di dati da inviare
            const int sendSize = (localHeight * localWidth * 3);

            // Invia solo la porzione di dati richiesta a destra
            for (int i = 0; i < size; i++)
                if (i != rank_first)
                    MPI_Isend(&frame.data[localHeight * localWidth * 3 * i], sendSize, MPI_CHAR, i, 0, comm2D, &request);
        }

        #pragma region Ricevi_Frame_Decodificato
            if (rank != rank_first){
                MPI_Isend(asciiArtPixelColor, localHeight * localWidth, sdl_color, rank_first, 0, comm2D, &request);
            }else{
                MPI_Status status;
                for (int i = 0; i < size; i++){
                    if (i != rank_first && i != rank_last) 
                        MPI_Irecv(&allAsciiArtPixelColor[localWidth * localHeight * i], localWidth * localHeight, sdl_color, i, 0, comm2D, &request);
                }

                if (rank_last != rank_first) 
                    MPI_Recv(&allAsciiArtPixelColor[localWidth * localHeight * rank_last], localWidth * localHeight, sdl_color, rank_last, 0, comm2D, &status);
            }
        #pragma endregion

        #pragma region Display_Frame
            if (rank == rank_first && operation_mode == GRAPHICS) {
                
                SDL_RenderClear(renderer);

                SDL_Rect destRect;
                destRect.w = PIXEL_SCALE;
                destRect.h = PIXEL_SCALE;

                for (int y = 0; y < ASCII_HEIGHT; y++) {
                    for (int x = 0; x < ASCII_WIDTH; x++) {
                        destRect.x = x * PIXEL_SCALE;
                        destRect.y = y * PIXEL_SCALE;

                        SDL_Color c = allAsciiArtPixelColor[y * ASCII_WIDTH + x];

                        SDL_SetTextureColorMod(asciiTextures[c.a], c.b, c.g, c.r);

                        SDL_RenderCopy(renderer, asciiTextures[c.a], NULL, &destRect);
                    }
                }

                SDL_RenderPresent(renderer);

            }

            if (rank == rank_first && i % 10 == 0){
                printf("Done %d frames out of %d\n", i, nFrames);
            }
        }
    #pragma endregion
}

int parseVideoConfig(const char* filename, int* op_mode, int* scaleSize) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return -1;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        // Remove newline character
        line[strcspn(line, "\n")] = '\0';

        // Find the key-value separator
        char* separator = strchr(line, '=');
        if (separator == NULL) {
            continue;
        }

        // Split the line into key and value
        *separator = '\0';
        char* fileKey = line;
        char* fileValue = separator + 1;

        // Trim leading and trailing whitespace from the key and value
        while (*fileKey == ' ' || *fileKey == '\t') {
            fileKey++;
        }
        while (*(fileValue - 1) == ' ' || *(fileValue - 1) == '\t') {
            *(fileValue - 1) = '\0';
            fileValue--;
        }

        // Compare the keys and assign values
        if (strcmp(fileKey, "mode") == 0) {
            *op_mode = atoi(fileValue);
        } else if (strcmp(fileKey, "scale_size") == 0) {
            *scaleSize = atoi(fileValue);
        }else if (strcmp(fileKey, "profiler") == 0){
            operation_mode = abs(atoi(fileValue)-1); //if profiles value is 1: 1-1 = 0 so profiler enabled, else abs(0-1) = 1, profiler disabled
        }else if (strcmp(fileKey, "video_path") == 0){
            strcpy(video_path, fileValue);
        }
    }

    fclose(file);
    return 0;
}


void profiler(int rank, int size){
    double starttime;
    
    operation_mode = 0;
    for (int i = 0; i < 10; ++i){

        if(rank==0){
            starttime=MPI_Wtime();
        }
        
        processFrames(rank, size);
        
        if (rank == 0) {
            double finaleTime = MPI_Wtime()-starttime;
            printf("Time: %2.5f\n", finaleTime);
            printf("Time per frame: %2.5fms, correct frame ms: %2.5fms\n", (finaleTime/nFrames) * 1000, (1.f/framerate) * 1000);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Con grafica\n");

    operation_mode = 1;
    init(rank, size); 
    for (int i = 0; i < 10; ++i){

        if(rank==0)
            starttime=MPI_Wtime();
        
        processFrames(rank, size);
        
        if (rank == 0) {
            double finaleTime = MPI_Wtime()-starttime;
            printf("Time: %2.5f\n", finaleTime);
            printf("Time per frame: %2.5fms, correct frame ms: %2.5fms\n", (finaleTime/nFrames) * 1000, (1.f/framerate) * 1000);
        }
    }
}


int main(int argc, char *argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("rank: %d\n", rank);

    if (parseVideoConfig("config.txt", &operation_mode, &PIXEL_SCALE)){
        printf("Impossible to parse the config file\n");
        return 0;
    }

    init(rank, size); 

    if (operation_mode == 0){
        profiler(rank, size);
    }else{
        double starttime=MPI_Wtime();

        processFrames(rank, size);

        double finaleTime = MPI_Wtime()-starttime;
        printf("Time: %2.5f\n", finaleTime);
        printf("Time per frame: %2.5fms, correct frame ms: %2.5fms\n", (finaleTime/nFrames) * 1000, (1.f/framerate) * 1000);
    }

    if (rank == rank_first) {
        destroySDL(window, renderer, font);
    }
    
    free(allAsciiArtIdx);
    free(asciiTextures);

    MPI_Finalize();
    return 0;
}


//things to install
//sudo apt install ffmpeg
//sudo apt install libavformat-dev
//sudo apt install libopencv-dev
//to compile it
//mpic++ main.c -o a -lSDL2 -lSDL2_ttf -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio
//mac os:
// export LIBRARY_PATH="$LIBRARY_PATH:/opt/homebrew/lib"
// mpic++ main.c -o a -lSDL2 -lSDL2_ttf -I/opt/homebrew/Cellar/opencv/4.8.1/include/opencv4 -I/opt/homebrew/Cellar/sdl2/2.28.3/include/SDL2 -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio -std=c++11 -O3

/*

640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 23.44451
Time per frame: 101.49140ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 24.95490
Time per frame: 108.02987ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 22.52514
Time per frame: 97.51142ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 22.13794
Time per frame: 95.83525ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 21.16549
Time per frame: 91.62551ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 21.48039
Time per frame: 92.98870ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 24.10132
Time per frame: 104.33474ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 21.15259
Time per frame: 91.56966ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 29.50400
Time per frame: 127.72293ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 23.37530
Time per frame: 101.19177ms, correct frame ms: 41.66667ms
Con grafica
Con grafica
640, 360
Time: 206.66555
Time per frame: 894.65604ms, correct frame ms: 41.66667ms
640, 360
Time: 173.10263
Time per frame: 749.36205ms, correct frame ms: 41.66667ms
640, 360
Time: 166.73530
Time per frame: 721.79782ms, correct frame ms: 41.66667ms
640, 360
Time: 158.69720
Time per frame: 687.00087ms, correct frame ms: 41.66667ms
640, 360
Time: 158.65587
Time per frame: 686.82194ms, correct frame ms: 41.66667ms
640, 360
Time: 163.55125
Time per frame: 708.01406ms, correct frame ms: 41.66667ms
640, 360
Time: 154.11591
Time per frame: 667.16846ms, correct frame ms: 41.66667ms
640, 360
Time: 154.85602
Time per frame: 670.37239ms, correct frame ms: 41.66667ms
640, 360
Time: 152.70466
Time per frame: 661.05915ms, correct frame ms: 41.66667ms
640, 360
Time: 155.53769
Time per frame: 673.32335ms, correct frame ms: 41.66667ms

*/

//n1
/*
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 12.86559
Time per frame: 55.69518ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 15.62100
Time per frame: 67.62336ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 8.82373
Time per frame: 38.19798ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 8.06285
Time per frame: 34.90410ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 7.98836
Time per frame: 34.58164ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 8.09360
Time per frame: 35.03723ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 8.92099
Time per frame: 38.61899ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 8.99403
Time per frame: 38.93520ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 9.18001
Time per frame: 39.74031ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 8.93494
Time per frame: 38.67938ms, correct frame ms: 41.66667ms
Con grafica
640, 360
Time: 81.31357
Time per frame: 352.00678ms, correct frame ms: 41.66667ms
640, 360
Time: 83.83648
Time per frame: 362.92847ms, correct frame ms: 41.66667ms
640, 360
Time: 85.09684
Time per frame: 368.38461ms, correct frame ms: 41.66667ms
640, 360
Time: 86.08430
Time per frame: 372.65932ms, correct frame ms: 41.66667ms
640, 360
Time: 85.21954
Time per frame: 368.91575ms, correct frame ms: 41.66667ms
640, 360
Time: 96.45111
Time per frame: 417.53728ms, correct frame ms: 41.66667ms
640, 360
Time: 95.94886
Time per frame: 415.36302ms, correct frame ms: 41.66667ms
640, 360
Time: 104.10433
Time per frame: 450.66810ms, correct frame ms: 41.66667ms
640, 360
Time: 114.00885
Time per frame: 493.54480ms, correct frame ms: 41.66667ms
640, 360
Time: 116.60699
Time per frame: 504.79218ms, correct frame ms: 41.66667ms
*/


//-o3 n1
/*
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 10.08736
Time per frame: 43.66823ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 8.32109
Time per frame: 36.02204ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 9.70571
Time per frame: 42.01608ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 12.37293
Time per frame: 53.56248ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 6.24588
Time per frame: 27.03846ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 6.19859
Time per frame: 26.83371ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 6.30770
Time per frame: 27.30606ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 6.17648
Time per frame: 26.73801ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 7.22914
Time per frame: 31.29497ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 7.87981
Time per frame: 34.11172ms, correct frame ms: 41.66667ms
Con grafica
640, 360
Time: 70.03825
Time per frame: 303.19589ms, correct frame ms: 41.66667ms
640, 360
Time: 64.95602
Time per frame: 281.19489ms, correct frame ms: 41.66667ms
640, 360
Time: 65.41953
Time per frame: 283.20143ms, correct frame ms: 41.66667ms
640, 360
Time: 65.19063
Time per frame: 282.21051ms, correct frame ms: 41.66667ms
640, 360
Time: 63.52727
Time per frame: 275.00982ms, correct frame ms: 41.66667ms
640, 360
Time: 67.40332
Time per frame: 291.78927ms, correct frame ms: 41.66667ms
640, 360
Time: 75.83705
Time per frame: 328.29892ms, correct frame ms: 41.66667ms
640, 360
Time: 76.67499
Time per frame: 331.92638ms, correct frame ms: 41.66667ms
640, 360
Time: 83.47335
Time per frame: 361.35650ms, correct frame ms: 41.66667ms
640, 360
Time: 85.41853
Time per frame: 369.77719ms, correct frame ms: 41.66667ms
*/

//-O3 n2
/*
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 8.67820
Time per frame: 37.56795ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 7.62961
Time per frame: 33.02860ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 7.59634
Time per frame: 32.88457ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 7.56356
Time per frame: 32.74270ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 7.28101
Time per frame: 31.51951ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 7.31535
Time per frame: 31.66820ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 7.57828
Time per frame: 32.80641ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 7.81151
Time per frame: 33.81604ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 7.49247
Time per frame: 32.43492ms, correct frame ms: 41.66667ms
640, 360
Done 0 frames out of 231
Done 10 frames out of 231
Done 20 frames out of 231
Done 30 frames out of 231
Done 40 frames out of 231
Done 50 frames out of 231
Done 60 frames out of 231
Done 70 frames out of 231
Done 80 frames out of 231
Done 90 frames out of 231
Done 100 frames out of 231
Done 110 frames out of 231
Done 120 frames out of 231
Done 130 frames out of 231
Done 140 frames out of 231
Done 150 frames out of 231
Done 160 frames out of 231
Done 170 frames out of 231
Done 180 frames out of 231
Done 190 frames out of 231
Done 200 frames out of 231
Done 210 frames out of 231
Done 220 frames out of 231
Done 230 frames out of 231
Time: 7.85866
Time per frame: 34.02018ms, correct frame ms: 41.66667ms
Con grafica
Con grafica
640, 360
Time: 84.69553
Time per frame: 366.64731ms, correct frame ms: 41.66667ms
640, 360
Time: 87.35997
Time per frame: 378.18168ms, correct frame ms: 41.66667ms
640, 360
Time: 87.68797
Time per frame: 379.60159ms, correct frame ms: 41.66667ms
640, 360
Time: 91.85092
Time per frame: 397.62303ms, correct frame ms: 41.66667ms
640, 360
Time: 95.72886
Time per frame: 414.41067ms, correct frame ms: 41.66667ms
640, 360
Time: 97.77429
Time per frame: 423.26534ms, correct frame ms: 41.66667ms
640, 360
Time: 124.54629
Time per frame: 539.16144ms, correct frame ms: 41.66667ms
640, 360
Time: 115.72205
Time per frame: 500.96126ms, correct frame ms: 41.66667ms
640, 360
Time: 111.74365
Time per frame: 483.73876ms, correct frame ms: 41.66667ms
640, 360
Time: 114.88664
Time per frame: 497.34475ms, correct frame ms: 41.66667ms
*/