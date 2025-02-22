using Cheese.Module;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
            resRollOutAI.adam = new(resRollOutAI.parameters(), 1E-4);
            resRollOutAI.load(RollName + ".dat");

            AITrainer.rollOutAI = resRollOutAI;

            string Train = Console.ReadLine();
            TrainTime = int.Parse(Train);
            for (int i = 0; i < TrainTime; i++)
            {
                Console.WriteLine(i);
                AITrainer.RolloutPlay();
                
            }
        }
    }
    
}
