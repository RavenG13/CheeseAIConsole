using System.Collections.Concurrent;
using TorchSharp;
using static TorchSharp.torch;

namespace ConsoleApp1.Support
{
    internal class DoubleBuffer
    {
        public object _inputLock = new();
        private ConcurrentQueue<Tensor> tensorsBuffer = new();
        private ConcurrentQueue<OutPut> outPutsBuffer = new();
        public OutPut PutIn(Tensor tensor)
        {
            lock (_inputLock)
            {
                tensorsBuffer.Enqueue(tensor);
            }

            OutPut outPut = new();

            outPut.OutCollection = new BlockingCollection<(Tensor output, Tensor value)>();
            outPutsBuffer.Enqueue(outPut);
            return outPut;
        }
        public void Forward()
        {
            Tensor input;
            lock (_inputLock)
            {
                Tensor[] tensors = tensorsBuffer.ToArray<Tensor>();
                input = torch.cat(tensors);
                tensorsBuffer.Clear();
            }

            (Tensor output, Tensor value) = (torch.pow(input,2),torch.ones(new long[] {3,1}));
            int index = 0;
            while (outPutsBuffer.Count != 0)
            {
                if (outPutsBuffer.TryDequeue(out OutPut result))
                {
                    Console.WriteLine(index);
                    result.OutCollection.TryAdd((output[index], value[index]));
                    index++;
                }
            }

        }

    }

    public class OutPut
    {
        public BlockingCollection<(Tensor output, Tensor value)>? OutCollection { get; set; }
        public (Tensor, Tensor) Output;
    }
}
