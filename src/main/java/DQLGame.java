import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.awt.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class DQLGame extends Thread {

    GameWorld world;
    DQLearner RLAI;

    double ExplorationEpsilon = 1d;
    float FutureDiscount = .99f;

    Visualizer grapher;
    Visualizer vizGame;

    int AveragePer = 10;
    int DQLNseed = 12345;
    int WorldGenSeed = 0;
    int MaxMovesPerEpisode;
    public DQLGame(){
        world = new GameWorld();
        MaxMovesPerEpisode = GameWorld.WorldSize * GameWorld.WorldSize;
    }
    void initDQLN(){
        int inputNeurons = GameWorld.WorldComplexity;
        int hiddenLayerNeurons = 150; //Mess around with this?
        MultiLayerConfiguration NNConf = new NeuralNetConfiguration.Builder()
                .seed(DQLNseed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.0005))   //Mess around with this value?
                .l2(0.001)
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(inputNeurons)
                        .nOut(hiddenLayerNeurons)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(hiddenLayerNeurons)
                        .nOut(hiddenLayerNeurons)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(hiddenLayerNeurons)
                        .nOut(GameWorld.ActionAmount)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.IDENTITY)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true).build();

        RLAI = new DQLearner(NNConf, 100000, FutureDiscount, ExplorationEpsilon, DQLNseed, 1024, 500, 1024, inputNeurons, GameWorld.ActionAmount);
        System.out.println("RLAI Intiated");
    }

    int playEpisode(int seed){
        System.out.println("Episode: " + world.EpisodeCount);
        if(RLAI.ExplorationEpsilon > 0) {
            RLAI.setExplorationEpsilon(RLAI.ExplorationEpsilon - 0.002);
        }
        System.out.println("e = " + RLAI.ExplorationEpsilon);
        float[][] currentMap = world.generateWorld(seed);
        world.shortPathEstimations[world.EpisodeCount] = world.shortestPathEstimation(currentMap);
        vizGame.addGamePanel(currentMap);
        world.printMap(currentMap);
        int timestep = 0;
        for(; timestep <= MaxMovesPerEpisode ; timestep++){
            world.setBufferFrame(currentMap);
            int action = RLAI.eGreedyAction(world.preprocessState(timestep), world.getActionMask(currentMap));
            float[][] nextMap = world.processMove(currentMap,action);
            float reward = world.getReward(nextMap);
            world.printMap(nextMap);
            try {
                System.out.println("delay(ms): " + vizGame.drawDelayms);
                vizGame.g.setState(nextMap);
                sleep(vizGame.drawDelayms);
            }catch(Exception e){
                System.out.println("Could not sleep");
            }
            if(reward == world.WinReward){
                RLAI.Qupdate(reward, null, world.getActionMask(nextMap));
                break;
            }else{
                RLAI.Qupdate(reward, world.preprocessState(timestep), world.getActionMask(nextMap));
                currentMap = nextMap;
            }
        }
        if(world.movesPerEpisode[world.EpisodeCount] <= world.shortPathEstimations[world.EpisodeCount]){
            world.totalWins++;
        }else{
            world.totalLosses++;
        }
        return (timestep +1); //Move 0 is move 1
    }

    void resetWorld(){
        world = new GameWorld();
    }

    void prepareVisuals(){
        grapher = new Visualizer();
        grapher.addPlotter(3,world.MaxEpisodes, MaxMovesPerEpisode, "Episode", "Moves");
        grapher.p.addGraph(world.MaxEpisodes ,false);
        grapher.makeVisible();
        vizGame = new Visualizer();
        vizGame.addFPSSlider();
        vizGame.makeVisible();
    }

    void visualizeAverages(){
        grapher.p.addGraph(world.MaxEpisodes/AveragePer, false);
        grapher.p.graphs[1].setColor(Color.gray);
        grapher.p.graphs[1].addLabel("Average");
        for(int i = 0; i < world.movesPerEpisode.length; i += AveragePer){
            int average = 0;
            for(int c = 0; c < AveragePer; c++){
                average += world.movesPerEpisode[c+i];
            }
            average /= AveragePer;
            System.out.println(i + "average: " + average);
            grapher.p.addPoint(1,i+(AveragePer/2), average);
        }
    }

    void visualizeEstimations(){
        grapher.p.addGraph(world.MaxEpisodes, false);
        grapher.p.graphs[2].setColor(Color.ORANGE);
        grapher.p.graphs[2].addLabel("Target");
        Visualizer vizzy = new Visualizer();
        vizzy.addPlotter(2, world.MaxEpisodes, MaxMovesPerEpisode, "Episode", "Moves");
        vizzy.p.setTitle("Unnecessary moves");
        vizzy.p.addGraph(world.MaxEpisodes,true);
        for(int i = 0; i < world.shortPathEstimations.length; i++){
            grapher.p.addPoint(2,i, world.shortPathEstimations[i]);
            vizzy.p.addPoint(0, i, (world.movesPerEpisode[i]-world.shortPathEstimations[i]));
            System.out.println(i +" Score " + (world.movesPerEpisode[i] - world.shortPathEstimations[i]));
        }
        vizzy.makeVisible();
    }

    void saveResults(String filename){
        try{
            String s = "Episode" + "\t" + "Moves" + "\t" + "WorldGenSeed: " + WorldGenSeed;

            for(int i = 0; i < world.movesPerEpisode.length; i++){
                s += System.lineSeparator() +(i+1) + "\t" + world.movesPerEpisode[i] + "\t" + (world.movesPerEpisode[i] - world.shortPathEstimations[i]);
            }
            Path path = Paths.get(filename);
            Files.write(path, s.getBytes());
            System.out.println(s);
        }catch(IOException e){
            System.out.println("ERROR writing in: " + filename);
        }
    }

    public static void main(String[] args){
        DQLGame game = new DQLGame();
        game.initDQLN();
        game.prepareVisuals();
        Random worldrng;
        if(game.WorldGenSeed == 0){
            worldrng = new Random();
            game.WorldGenSeed = worldrng.nextInt();
            worldrng.setSeed(game.WorldGenSeed);
        }else{
            worldrng = new Random(game.WorldGenSeed);
        }
        for(; game.world.EpisodeCount < game.world.MaxEpisodes; game.world.EpisodeCount++){
            int episodeMoves = game.playEpisode(worldrng.nextInt());
            game.world.movesPerEpisode[game.world.EpisodeCount] = episodeMoves;
            game.grapher.p.addPoint(0,game.world.EpisodeCount+1, episodeMoves);
        }
        game.visualizeAverages();
        game.visualizeEstimations();
        game.saveResults("Results.txt");
        System.out.println("Total wins: " + game.world.totalWins + ", Losses: " + game.world.totalLosses);
    }

}
