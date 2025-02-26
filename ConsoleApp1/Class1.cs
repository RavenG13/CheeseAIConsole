using Cheese.Module;
using TorchSharp;

namespace ConsoleApp1
{
    public static class Global
    {
        public const int SIZE = 15;
        private static int TrainTime;
        public static string RollName = "rollout";
        public static void Main(string[] args)
        {
            ResRollOutAI resRollOutAI = new(RollName);
            resRollOutAI.to(DeviceType.CUDA);
            resRollOutAI.load(RollName + ".dat");
            resRollOutAI.adam = new(resRollOutAI.parameters(), 1E-4);
            AITrainer.rollOutAI = resRollOutAI;

            AITrainer.alphaAI = new("", 15, 7);
            AITrainer.alphaAI.to(DeviceType.CUDA);
            AITrainer.alphaAI.load("./ModuleSave/1.dat");
            AITrainer.alphaAI.optimizer = new(AITrainer.alphaAI.parameters(), 5E-5);
            
            string Train = Console.ReadLine();
            
            try { TrainTime = int.Parse(Train); }
            catch (Exception ex) { TrainTime = 1000; }
            Console.Write(TrainTime);

            for (int i = 0; i < TrainTime; i++)
            {
                Console.WriteLine("study_time="+i);
                AITrainer.SelfPlay();

            }
        }
    }

}
