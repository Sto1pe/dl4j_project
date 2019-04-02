
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class GameWorld {
    static int WorldSize = 3; // = Width = Length. Actual size = this*this. Square world
    static int WorldLayers = 2; //Layers of complexity(amount of different objects) 3: Agent, Goal, (Walls),(Enemies),
    static int WorldComplexity = WorldSize* WorldSize * WorldLayers +1; //total info + info about time (therefore +1)
    static int ActionAmount = 4;
    static float WinReward = WorldSize * WorldSize +1;  //Why this value??
    int totalWins = 0;
    int totalLosses = 0;
    int[] movesPerEpisode;
    int[] shortPathEstimations;
    int EpisodeCount = 0;
    int MaxEpisodes = 4000;
    float[][] BufferFrame;

    public GameWorld(){
        movesPerEpisode = new int[MaxEpisodes];
        shortPathEstimations = new int[MaxEpisodes];
    }

    float[][] generateWorld(int seed){
        Random rng;
        rng = new Random(seed);

        int agent = rng.nextInt(WorldSize*WorldSize);
        int goal = rng.nextInt(WorldSize*WorldSize);
        while(goal == agent){
            goal = rng.nextInt(WorldSize*WorldSize);
        }
        float[][] map = new float[WorldSize][WorldSize];
        for (int i = 0; i < WorldSize * WorldSize; i++)
            map[i / WorldSize][i % WorldSize] = 0;
        map[goal / WorldSize][goal % WorldSize] = -1;
        map[agent / WorldSize][agent % WorldSize] = 1;

        return map;
    }

    int findOnMap(float[][] Map, int identifier){
        for(int i = 0; i < WorldSize* WorldSize; i++){
            if(Map[i/WorldSize][i % WorldSize] == identifier){
                return i;
            }
        }
        System.out.println("ERROR: Object with identifier: " + identifier + " couldn't be found");
        return -1;
    }

    int[] getActionMask(float[][] Map){
        int[] acceptedActions = {1, 1, 1, 1};   //Up,down,left,right all possible
        int agent = findOnMap(Map, 1);
        if (agent < WorldSize) {
            acceptedActions[0] = 0;
        }
        if (agent >= WorldSize * WorldSize - WorldSize) {
            acceptedActions[1] = 0;
        }
        if (agent % WorldSize == 0) {
            acceptedActions[2] = 0;
        }
        if (agent % WorldSize == WorldSize - 1) {
            acceptedActions[3] = 0;
        }
        return acceptedActions;
    }

    float[][] processMove(float[][] Map, int action){
        float[][] newMap = Map.clone();
        int agent = findOnMap(Map, 1);
        newMap[agent/WorldSize][agent%WorldSize] = 0;

        if (action == 0) {      //UP
            if (agent - WorldSize >= 0)
                newMap[(agent - WorldSize) / WorldSize][agent % WorldSize] = 1;
            else {
                System.out.println("ERROR: Bad move: can't go UP");
            }
        } else if (action == 1) {   //Down
            if (agent + WorldSize < WorldSize * WorldSize)
                newMap[(agent + WorldSize) / WorldSize][agent % WorldSize] = 1;
            else {
                System.out.println("ERROR: Bad move: can't go DOWN");
            }
        } else if (action == 2) {   //Left
            if ((agent % WorldSize) - 1 >= 0)
                newMap[agent / WorldSize][(agent % WorldSize) - 1] = 1;
            else {
                System.out.println("ERROR: Bad move: can't go LEFT");
            }
        } else if (action == 3) {   //Right
            if ((agent % WorldSize) + 1 < WorldSize)
                newMap[agent / WorldSize][(agent % WorldSize) + 1] = 1;
            else {
                System.out.println("ERROR: Bad move: can't go RIGHT");
            }
        }
        return newMap;
    }

    INDArray preprocessState(int timestep){
        float[] processedState = new float[WorldComplexity];
        for(int a = 0; a < WorldSize; a++){
            for(int b = 0; b < WorldSize; b++){
                if(BufferFrame[a][b] == -1){
                    processedState[a * WorldSize+ b] = 1;
                }else{
                    processedState[a*WorldSize +b] = 0;
                }
                if(BufferFrame[a][b] == 1){
                    processedState[WorldSize*WorldSize + a*WorldSize + b] = 1;
                }else{
                    processedState[WorldSize*WorldSize + a*WorldSize + b] = 0;
                }
            }
        }
        processedState[WorldSize*WorldSize*2] = timestep;
        return Nd4j.create(processedState);
    }

    void setBufferFrame(float[][] frame){
        BufferFrame = frame;
    }

    float getReward(float[][] nextMap){
        if(findOnMap(nextMap, -1) == -1){   //If it can't find goal, player is on goal, give win reward.
            return WinReward;
        }else{
            return -1f;
        }

    }

    void printMap(float[][] map){
        for(int y = 0; y < WorldSize; y++){
            for(int x = 0; x < WorldSize; x++){
                System.out.print((int) map[y][x]);
            }
            System.out.println();
        }
        System.out.println();
    }

    int shortestPathEstimation(float[][] map){  //Manhattan Distance Heuristics
        int agentX = 0, agentY = 0,goalX = 0,goalY = 0;
        for(int y = 0; y < map.length; y++){
            for(int x = 0; x < map[0].length; x++){
                if(map[y][x] == 1){
                    agentX = x;
                    agentY = y;
                }
                if(map[y][x] == -1){
                    goalX = x;
                    goalY = y;
                }
            }
        }
        return Math.abs(agentX-goalX) + Math.abs(agentY - goalY);
    }

}
