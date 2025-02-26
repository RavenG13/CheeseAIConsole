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
            outPut.mutex = new Mutex();
            outPut.mutex.WaitOne();
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

            (Tensor output, Tensor value) = (torch.pow(input,2),torch.ones(1));
            while (outPutsBuffer.Count != 0)
            {
                if (outPutsBuffer.TryDequeue(out OutPut result))
                {
                    result.Output = (output, value);
                    result.mutex.ReleaseMutex();
                }
            }

        }

    }

    struct OutPut
    {
        public Mutex mutex;
        public (Tensor, Tensor) Output;
    }
}
