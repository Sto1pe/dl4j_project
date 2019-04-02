import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Random;
import java.util.List;

public class DQLearner {

    int ExpMemoryCapacity;
    List<ReplayMemory> ExpReplay;
    double ExplorationEpsilon;
    float FutureDiscount;
    MultiLayerNetwork DQLN, targetDQLN;
    int BatchSize;
    int UpdateFreq;
    int UpdateCounter;
    int ReplayStartSize;
    Random rng;
    int InputNeurons;
    int NumActions;
    INDArray LastInput;
    int LastAction;

    DQLearner(MultiLayerConfiguration NNConf, int expMemoryCapacity, float futureDiscount, double epsilon, int rngSeed, int batchSize, int updateFreq, int replayStartSize, int inputNeurons, int numActions){
        DQLN = new MultiLayerNetwork(NNConf);
        DQLN.init();

        targetDQLN = new MultiLayerNetwork(NNConf);
        targetDQLN.init();
        targetDQLN.setParams(DQLN.params());

        ExpMemoryCapacity = expMemoryCapacity;
        ExplorationEpsilon = epsilon;
        FutureDiscount = futureDiscount;
        rng = rngSeed == 0 ? new Random() : new Random(rngSeed);        //a seed of 0 is no seed
        BatchSize = batchSize;
        ExpReplay = new ArrayList<ReplayMemory>();
        UpdateFreq = updateFreq;
        UpdateCounter = 0;
        ReplayStartSize = replayStartSize;
        InputNeurons = inputNeurons;
        NumActions = numActions;
    }

    int getMaxQAction(INDArray NNoutput, int[] actionMask){
        int i = 0;
        while(actionMask[i] == 0) i++; //Skips ignored actions before first true value(defined by mask)

        float maxQ = NNoutput.getFloat(i);
        int maxQindex = i;      //Looping through the remaining values to find true maxQ
        for(; i < NNoutput.size(1); i++){
            if(NNoutput.getFloat(i) > maxQ && actionMask[i] == 1){
                maxQ = NNoutput.getFloat(i);
                maxQindex = i;
            }
        }
        return maxQindex;
    }

    float getMaxQ(INDArray NNoutput, int[] actionMask){
        int i = 0;
        while(actionMask[i] == 0) i++; //Skips ignored actions before first true value(defined by mask)

        float maxQ = NNoutput.getFloat(i);  //Looping through the remaining values to find true maxQ
        for(; i < NNoutput.size(1); i++){
            if(NNoutput.getFloat(i) > maxQ && actionMask[i] == 1){
                maxQ = NNoutput.getFloat(i);
            }
        }
        return maxQ;
    }

    int eGreedyAction(INDArray inputs, int[] actionMask){
        LastInput = inputs;
        INDArray output = DQLN.output(inputs);
        System.out.println(output + " ");

        for(int i : actionMask){
            System.out.print(" " + i);
        }

        if(ExplorationEpsilon > rng.nextDouble()){
            do{
                LastAction = rng.nextInt((int) output.size(1));
            }while (actionMask[LastAction] == 0);
            System.out.println("Chose random action: "+ LastAction);
            return LastAction;
        }else{
            LastAction = getMaxQAction(output, actionMask);
            System.out.println("Chose DQL action: " + LastAction);
            return LastAction;
        }
    }

    void Qupdate(float reward, INDArray nextInputs, int[] nextActionMask){
        addReplay(reward, nextInputs, nextActionMask);
        if(ReplayStartSize < ExpReplay.size()){ //Start training of the DQNs when enough experience is there
            trainDQNs(BatchSize);
        }
        UpdateCounter++;
        if(UpdateCounter == UpdateFreq){
            UpdateCounter = 0;
            System.out.println("Reconciling Networks");
            reconcileNN();
        }
    }
    //TODO: Check through to fully understand each process.
    void trainDQNs(int batchSize){
        ReplayMemory[] replays = getMemoryBatch(batchSize);
        INDArray CurrInputs = combineInputs(replays);
        INDArray TargetInputs = combineNextInputs(replays);

        INDArray CurrOutputs = DQLN.output(CurrInputs);
        INDArray TargetOutputs = targetDQLN.output(TargetInputs);

        float[] y = new float[replays.length];
        for(int i = 0; i < y.length; i++){
            int[] ind = {i, replays[i].Action};
            float futureReward = 0;
            if(replays[i].NextInput != null){
                futureReward = getMaxQ(TargetOutputs.getRow(i), replays[i].NextActionMask);
            }
            float targetReward = replays[i].Reward + (FutureDiscount * futureReward);
            CurrOutputs.putScalar(ind, targetReward);
        }

        DQLN.fit(CurrInputs,CurrOutputs);
    }

    //**************** HELPERFUNCTIONS ****************\\

    void reconcileNN(){
        targetDQLN.setParams(DQLN.params());
    }

    void setExplorationEpsilon(double epsilon){ExplorationEpsilon = epsilon;}

    INDArray combineInputs(ReplayMemory[] replays){
        INDArray combInputs = Nd4j.create(replays.length, InputNeurons);
        for(int i = 0; i < replays.length; i++){
            combInputs.putRow(i,replays[i].Input);
        }
        return combInputs;
    }

    INDArray combineNextInputs(ReplayMemory[] replays){
        INDArray combInputs = Nd4j.create(replays.length, InputNeurons);
        for(int i = 0; i < replays.length; i++){
            if(replays[i].NextInput != null)
                combInputs.putRow(i, replays[i].NextInput);;
        }
        return combInputs;
    }

    ReplayMemory[] getMemoryBatch(int batchSize){
        int size = ExpReplay.size() < BatchSize ? ExpReplay.size() : BatchSize;
        ReplayMemory[] batch = new ReplayMemory[size];
        for(int i = 0; i < size; i++){
            batch[i] = ExpReplay.get(rng.nextInt(ExpReplay.size()));
        }
        return batch;
    }

    void addReplay(float reward, INDArray nextInput, int[] nextActionMask){
        if(ExpReplay.size() >= ExpMemoryCapacity){
            ExpReplay.remove(rng.nextInt(ExpReplay.size()));
        }
        ExpReplay.add(new ReplayMemory(LastInput, LastAction, reward, nextInput,nextActionMask));
    }
}
