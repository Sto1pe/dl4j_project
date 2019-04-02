package Learn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

public class GridWorld extends Thread{
    DeepQNetwork RLNet;
    int size = 5;
    int episodeAmount = 1000;
    //int scale = 3;

    int NNSeed = 12345;
    int rngSeed = 123;

    float FrameBuffer[][];
    int[] movesPerEpisode = new int[episodeAmount];
    int[] episodeCount = new int[episodeAmount];

    // Network initialization
    void networkConstruction() {
        int InputLength = size * size * 2 + 1;
        int HiddenLayerCount = 150;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(NNSeed)    //Random number generator seed for improved repeatability. Optional.
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .l2(0.001) // l2 regularization on all layers
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(InputLength)
                        .nOut(HiddenLayerCount)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(HiddenLayerCount)
                        .nOut(HiddenLayerCount)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(2,new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(HiddenLayerCount)
                        .nOut(4) // for 4 possible actions
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.IDENTITY)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true).build();

        RLNet = new DeepQNetwork(conf, 100000, .99f, 1d, 1024, 500, 1024, InputLength, 4, rngSeed);
    }

    Random rand = new Random();

    // Generate the GridMap
    float[][] generateGridMap() {
        int agent = rand.nextInt(size * size);
        int goal = rand.nextInt(size * size);
        while (goal == agent)
            goal = rand.nextInt(size * size);
        float[][] map = new float[size][size];
        for (int i = 0; i < size * size; i++)
            map[i / size][i % size] = 0;
        map[goal / size][goal % size] = -1;
        map[agent / size][agent % size] = 1;
        return map;
    }

    // Calculate the position of agent
    int calcAgentPos(float[][] Map) {
        int x = -1;
        for (int i = 0; i < size * size; i++) {
            if (Map[i / size][i % size] == 1)
                return i;
        }
        return x;
    }

    // Calculate the position of goal
    int calcGoalPos(float[][] Map) {
        int x = -1;
        for (int i = 0; i < size * size; i++) {
            if (Map[i / size][i % size] == -1)
                return i;
        }
        return x;
    }

    // Get action mask
    int[] getActionMask(float[][] CurrMap) {
        int retVal[] = { 1, 1, 1, 1 };

        int agent = calcAgentPos(CurrMap);
        if (agent < size)
            retVal[0] = 0;
        if (agent >= size * size - size)
            retVal[1] = 0;
        if (agent % size == 0)
            retVal[2] = 0;
        if (agent % size == size - 1)
            retVal[3] = 0;

        return retVal;
    }

    // Show guidance move to agent
    float[][] doMove(float[][] CurrMap, int action) {
        float nextMap[][] = new float[size][size];
        for (int i = 0; i < size * size; i++)
            nextMap[i / size][i % size] = CurrMap[i / size][i % size];

        int agent = calcAgentPos(CurrMap);
        nextMap[agent / size][agent % size] = 0;

        if (action == 0) {      //UP
            if (agent - size >= 0)
                nextMap[(agent - size) / size][agent % size] = 1;
            else {
                System.out.println("Bad Move");
                System.exit(0);
            }
        } else if (action == 1) {   //Down
            if (agent + size < size * size)
                nextMap[(agent + size) / size][agent % size] = 1;
            else {
                System.out.println("Bad Move");
                System.exit(0);
            }
        } else if (action == 2) {   //Left
            if ((agent % size) - 1 >= 0)
                nextMap[agent / size][(agent % size) - 1] = 1;
            else {
                System.out.println("Bad Move");
                System.exit(0);
            }
        } else if (action == 3) {   //Right
            if ((agent % size) + 1 < size)
                nextMap[agent / size][(agent % size) + 1] = 1;
            else {
                System.out.println("Bad Move");
                System.exit(0);
            }
        }
        return nextMap;
    }

    // Compute reward for an action
    float calcReward(float[][] CurrMap, float[][] NextMap) {
        int newGoal = calcGoalPos(NextMap);

        if (newGoal == -1)
            return size * size + 1;

        return -1f;
    }

    void addToBuffer(float[][] nextFrame) {
        FrameBuffer = nextFrame;
    }

    INDArray flattenInput(int TimeStep) {
        float flattenedInput[] = new float[size * size * 2 + 1];
        for (int a = 0; a < size; a++) {
            for (int b = 0; b < size; b++) {
                if (FrameBuffer[a][b] == -1)
                    flattenedInput[a * size + b] = 1;
                else
                    flattenedInput[a * size + b] = 0;
                if (FrameBuffer[a][b] == 1)
                    flattenedInput[size * size + a * size + b] = 1;
                else
                    flattenedInput[size * size + a * size + b] = 0;
            }
        }
        flattenedInput[size * size * 2] = TimeStep;
        return Nd4j.create(flattenedInput);
    }

    void printGrid(float[][] Map) {
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                System.out.print((int) Map[x][y]);
            }
            System.out.println(" ");
        }
        System.out.println(" ");
    }

    public static void main(String[] args) {
        GridWorld grid = new GridWorld();
        grid.networkConstruction();
        for(int c = 0; c < grid.episodeAmount; c++){
            grid.episodeCount[c] = c;
        }
        Visualizer viz = new Visualizer();
        viz.addPlot(grid.episodeAmount, 100, "Episode", "Moves");
        viz.makeVisible();
        Visualizer vizGame = new Visualizer();
        vizGame.addFPSSlider();
        vizGame.makeVisible();
        for (int m = 0; m < grid.episodeAmount; m++) {
            System.out.println("Test Episode: " + m);
            if(grid.RLNet.Epsilon > 0.1){
                grid.RLNet.SetEpsilon(grid.RLNet.Epsilon-0.01);
            }
            float CurrMap[][] = grid.generateGridMap();
            vizGame.addGamePanel(CurrMap);

            grid.FrameBuffer = CurrMap;
            int t = 0;

            for (int i = 0; i < 100; i++) {
                grid.printGrid(CurrMap);
                int a = grid.RLNet.getAction(grid.flattenInput(t), grid.getActionMask(CurrMap));

                float NextMap[][] = grid.doMove(CurrMap, a);
                float r = grid.calcReward(CurrMap, NextMap);
                grid.addToBuffer(NextMap);
                t++;

                try {
                    System.out.println("delay(ms): " + vizGame.drawDelayms);
                    vizGame.g.setState(NextMap);
                    sleep(vizGame.drawDelayms);
                }catch(Exception e){
                    System.out.println("Could not sleep");
                }

                if (r == grid.size * grid.size + 1) {
                    grid.RLNet.observeReward(r, null, grid.getActionMask(NextMap));
                    break;
                }

                grid.RLNet.observeReward(r, grid.flattenInput(t), grid.getActionMask(NextMap));
                CurrMap = NextMap;
            }
            grid.movesPerEpisode[m] = t;
            viz.p.addPoint(m, t);
        }

        Visualizer viz2 = new Visualizer();
        viz2.addPlot(100, 100, "Episode", "Moves");
        viz2.p.setTitle("Averages per 10");
        for(int i = 0; i < grid.movesPerEpisode.length; i += 10){
            int average = 0;
            for(int c = 0; c < 10; c++){
                average += grid.movesPerEpisode[c+i];
            }
            average /= 10;
            System.out.println(i + "average: " + average);
            viz2.p.addPoint(i/10, average);
        }
        viz2.makeVisible();
    }
}
