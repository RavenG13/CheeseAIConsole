using Cheese.Module;
using ConsoleApp1.Support;
using TorchSharp;
using static System.Net.Mime.MediaTypeNames;
using static TorchSharp.torch;

namespace ConsoleApp1
{
    public static class Global
    {
        public const int SIZE = 15;
        private static int TrainTime;
        public static string RollName = "rollout";
        public static void Test()
        {
            // 创建 DoubleBuffer 实例
            DoubleBuffer db = new DoubleBuffer();

            // 创建一些测试张量
            Tensor tensor1 = torch.tensor(new long[] { 1, 2 });
            Tensor tensor2 = torch.tensor(new long[] { 1, 2 });
            Tensor tensor3 = torch.tensor(new long[] { 1, 2 });

            // 添加张量到缓冲区
            OutPut[] outputs = new OutPut[3];
            outputs[0] = db.PutIn(tensor1);
            //outputs[1] = db.PutIn(tensor2);
            //outputs[2] = db.PutIn(tensor3);

            // 调用 Forward 方法
            Console.WriteLine(outputs[0].GetHashCode());
            db.Forward();

            // 验证结果
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i].mutex.WaitOne(); // 等待结果被处理
                (Tensor output, Tensor value) = outputs[i].Output;

                // 验证计算结果
                Tensor expectedOutput = torch.pow(tensor1 + i, 2); // 假设 forward 方法的输出是平方运算
                Tensor expectedValue = torch.ones(1);

                // 比较张量是否相等
                Console.WriteLine(torch.all(torch.eq(output, expectedOutput)).item<bool>());
                Console.WriteLine(torch.all(torch.eq(value, expectedValue)).item<bool>());
            }
        }
        public static void Main(string[] args)
        {
            //Test();
            ResRollOutAI resRollOutAI = new(RollName);
            resRollOutAI.to(DeviceType.CUDA);
            resRollOutAI.load(RollName + ".dat");
            resRollOutAI.adam = new(resRollOutAI.parameters(), 1E-4);
            AITrainer.rollOutAI = resRollOutAI;

            AITrainer.alphaAI = new("", 15, 7);
            AITrainer.alphaAI.to(DeviceType.CUDA);
            //AITrainer.alphaAI.load("./ModuleSave/1.dat");
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
