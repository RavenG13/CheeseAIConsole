﻿using ConsoleApp1;
using TorchSharp;
using static TorchSharp.torch;


public class Node
{
    public Mutex mutex {  get; private set; } = new Mutex();
    public Node Parent { get; private set; }
    public float ThreadLoss { get; set; }
    public Node[,] Children;
    private int _visitCount { get; set; }
    private double _valueSum;
    public double PriorP
    {
        get; set;
    }


    public Node(Node parent = null, double priorP = 1.3)
    {
        Parent = parent;
        PriorP = priorP;
        _visitCount = 0;
    }

    public double Value => _visitCount == 0 ? 0 : _valueSum / _visitCount;

    private void Update(double value)
    {
        _visitCount++;
        _valueSum += value;
        ThreadLoss = 0;
    }

    /// <summary>
    /// 更新自己叶子价值和父叶子价值
    /// </summary>
    /// <param name="leafValue"></param>
    /// <param name="negate"></param>
    public void UpdateRecursive(double leafValue, bool negate = true)
    {
        Update(leafValue);
        if (IsRoot()) return;
        Parent.UpdateRecursive(negate ? -leafValue : leafValue, negate);
    }

    public bool IsLeaf() => Children is null;
    public bool IsRoot() => Parent == null;
    public int VisitCount => _visitCount;
    public Node step(int[] pos)
    {
        return Children[pos[0], pos[1]];
    }
}
public class MCTS
{
    static float _pbCBase = 1.1f;

    private nn.Module<Tensor, (Tensor, Tensor)> module;
    private const int NumSimulations = 600;
    public MCTS(nn.Module<Tensor, (Tensor, Tensor)> module)
    {
        this.module = module;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="node"></param>
    /// <param name="alphaGo"></param>
    /// <param name="env"></param>
    /// <returns>叶节点价值</returns>
    public float ExpandLeafNode(Node node, Env env)
    {
        float LeafValue = 0;
        float[] ActionProbsArray;

        AITrainer.EnvMakeForwardTensor(env, out Tensor input, out Tensor all_Reshape_Input);

        (Tensor ActionProbs, Tensor leaf_value) = SelfForward(all_Reshape_Input);

        ActionProbs = ActionProbs.exp();

        ActionProbsArray = ActionProbs.data<float>().ToArray();

        LeafValue = leaf_value.item<float>();

        if (env.IsEnd().Item2 != 2) return LeafValue;

        MakeRandom(env, ActionProbsArray);
        node.Children = new Node[Global.SIZE, Global.SIZE];
        for (int i = 0; i < Global.SIZE; i++)
        {
            for (int j = 0; j < Global.SIZE; j++)
            {
                node.Children[i, j] = new Node(node, ActionProbsArray[i * Global.SIZE + j]);
            }
        }
        return LeafValue;
    }
    protected virtual (Tensor Act, Tensor LeafValue) SelfForward(Tensor all_Reshape_Input)
    {
        return module.forward(all_Reshape_Input);
    }
    private static void MakeRandom(Env env, float[] ActionProbsArray)
    {
        Random random = new Random();
        (int[], double) Score = (new int[] { 0, 0 }, -1);
        for (int i = 0; i < Global.SIZE; i++)
        {
            for (int j = 0; j < Global.SIZE; j++)
            {
                if (env.ToGameState().HasPiece(i, j))
                {
                    ActionProbsArray[i * Global.SIZE + j] = -10;
                    continue;
                }
                double value = random.NextDouble();
                if (value > Score.Item2) { Score = (new int[] { i, j }, value); }
            }
        }

        ActionProbsArray[Global.SIZE * Score.Item1[0] + Score.Item1[1]] += 0.3f;
    }
    public void Simulate(Node node, Env env)
    {
        Node root = node;
        Node Leaf = node;
        Env envCopy = env.Clone();
        Mutex mutex = null;

        SelectLeaf(ref Leaf, ref envCopy,ref mutex);

        (int[] pos, byte Winner) = envCopy.IsEnd();
        bool IsDone = Winner != 2;
        float LeafValue;

        if (!IsDone) { LeafValue = ExpandLeafNode(Leaf, envCopy); }
        else
        {
            LeafValue = envCopy.Player == envCopy.IsEnd().Item2 ? 1 : -1;
        }
        mutex.ReleaseMutex();
        //Console.WriteLine(Leaf.GetHashCode());
        //Console.WriteLine(Thread.GetCurrentProcessorId());
        Leaf.UpdateRecursive(-LeafValue);

    }

    protected static void SelectLeaf(ref Node Leaf, ref Env envCopy,ref Mutex mutex)
    {
        Node node;
        Leaf.mutex.WaitOne();

        while (!Leaf.IsLeaf())
        {
            (int[] Act, node) = SelectChild(Leaf);
            node.mutex.WaitOne();

            if (Act is null) {
                Leaf = node;
                mutex = node.mutex;
                break; 
            }

            Leaf.ThreadLoss = -10;
            envCopy = envCopy.Step(Act);
            Leaf.mutex.ReleaseMutex();
            Leaf = node;
        }
        mutex = Leaf.mutex;
    }

    /// <summary>
    /// 计算子节点的UCB值
    /// </summary>
    /// <param name="parent"></param>
    /// <param name="Child"></param>
    /// <returns></returns>
    public static float UcbScore(Node parent, Node Child)
    {
        double pbC = _pbCBase * Child.PriorP * Math.Sqrt(parent.VisitCount) / (Child.VisitCount + 1);
        return (float)(pbC + Child.Value) + Child.ThreadLoss;
    }
    /// <summary>
    /// 通过UCB分数选择叶节点
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    public static (int[], Node) SelectChild(Node node)
    {
        float BestScore = -999;
        int[] action = null;
        Node Child = null;

        for (int i = 0; i < Global.SIZE; i++)
        {
            for (int j = 0; j < Global.SIZE; j++)
            {
                float Ucb = UcbScore(node, node.Children[i, j]);
                if (Ucb > BestScore)
                {
                    action = new int[] { i, j };
                    Child = node.Children[i, j];
                    BestScore = Ucb;
                }
            }
        }
        if (Child == null) Child = node;
        return (action, Child);
    }

    public Tensor GetNextAction(Env env, Node root)
    {
        AITrainer.alphaAI.eval();

        torch.set_grad_enabled(false);
        
        int[] action = new int[2];

        if (root.Children is null) ExpandLeafNode(root, env);

        Task[] tasks = new Task[4];
        tasks[0] = Task.Run(() => Simulate(root, env));
        tasks[1] = Task.Run(() => Simulate(root, env));
        tasks[2] = Task.Run(() => Simulate(root, env));
        tasks[3] = Task.Run(() => Simulate(root, env));
        for (int i = root.VisitCount; i < NumSimulations; i++)
        {
            int x = Task.WaitAny(tasks);
            tasks[x] = Task.Run(() => Simulate(root, env));
        }


        float[,] ActionProbsArray = new float[Global.SIZE, Global.SIZE];
        for (int i = 0; i < Global.SIZE; i++)
        {
            for (int j = 0; j < Global.SIZE; j++)
            {
                ActionProbsArray[i, j] = (float)root.Children[i, j].VisitCount / (float)root.VisitCount;
            }
        }
        
        torch.Tensor ActionProbs = torch.tensor(ActionProbsArray);
        return ActionProbs.alias();
    }

}
public class Env
{
    public Env Parent { get; private set; }
    public Tensor TensorToLearn;
    private GameState gameState;
    public int Player { get; private set; }
    public Env() { gameState = new GameState(); Player = 0; }
    public Env(GameState gameState, int player, Env parent) { this.gameState = gameState; Player = player; Parent = parent; }
    public Env Clone()
    {
        Env env = new Env(gameState.Clone(), Player, this);
        env.Parent = this.Parent;
        return env;
    }
    public Env Step(int[] Action)
    {
        GameState NextGame = gameState.Clone();
        NextGame.Place(Player, Action);
        int player = Player == 0 ? 1 : 0;
        return new Env(NextGame, player, this);
    }
    public Tensor ToTensor()
    {
        return gameState.ToTensor();
    }
    public (int[], byte) IsEnd()
    {
        return gameState.IsEnd();
    }
    public override string ToString()
    {
        string i = gameState.Show();
        return i;
    }
    public string ShowEnv()
    {
        string output = "GameEnv\n";
        for (int i = 0; i < Global.SIZE; i++)
        {
            for(int j = 0;j<Global.SIZE;j++)
            {
                if (gameState.HasPiece(0, new int[] { i, j }))
                {
                    output += "0";
                }
                else if(gameState.HasPiece(1, new int[] { i, j }))
                {
                    output += "x";
                }
                else
                {
                    output += "-";
                }
            }
            output += "\n";
        }
        return output;
    }
    public Env GetRoot()
    {
        Env root = this;
        while (root.Parent != null)
        {
            root = root.Parent;
        }
        return root;
    }
    public GameState ToGameState()
    {
        return this.gameState;
    }
}
public class PureRollOutMcts : RollOutMCTS
{
    public PureRollOutMcts() : base(null, 3600)
    {

    }
    protected override (Tensor Act, Tensor LeafValue) SelfForward(Tensor all_Reshape_Input)
    {
        return (torch.nn.functional.log_softmax(torch.rand(new long[] { 1, Global.SIZE * Global.SIZE }), 1), torch.zeros(1));
    }
}

public class RollOutMCTS : MCTS
{
    private readonly int _threads = 4;
    protected readonly int RollOutTimes;
    private readonly nn.Module<Tensor, Tensor> RollAI;
    public RollOutMCTS(nn.Module<Tensor, Tensor> RollAI, int RollOutTimes = 600) : base(null)
    {
        this.RollAI = RollAI;
        this.RollOutTimes = RollOutTimes;
    }

    public float RollOut(Node root, Env env)
    {
        Random random = new Random();
        Env env1 = env.Clone();

        for (int i = 0; i < Global.SIZE * Global.SIZE; i++)
        {
            if (env1.IsEnd().Item2 != 2)
            { break; }
            double MaxRandom = -1;
            int[] Pos = new int[2];
            for (int x = 0; x < GameState.SIZE; x++)
            {
                for (int y = 0; y < GameState.SIZE; y++)
                {
                    if (env1.ToGameState().HasPiece(x, y)) { continue; }
                    double Value = random.NextDouble();
                    if (Value > MaxRandom) { Pos[0] = x; Pos[1] = y; MaxRandom = Value; }
                }
            }
            env1 = env1.Step(Pos);
        }
        byte Winner = env1.IsEnd().Item2;
        float LeafValue = Winner == env.Player ? 1f : -1f;
        return -LeafValue;
    }
    protected override (Tensor Act, Tensor LeafValue) SelfForward(Tensor all_Reshape_Input)
    {
        Tensor tensor = RollAI.forward(all_Reshape_Input.to(CUDA));
        return (tensor.to(CPU), torch.zeros(1));
    }

    public Tensor GetNextAction(Env env)
    {

        Node root = new Node();
        torch.set_grad_enabled(false);
        ExpandLeafNode(root, env);


        (Node node, Env env)[] LeafNodes = new (Node, Env)[_threads];
        for (int j = 0; j < _threads; j++)
        {
            LeafNodes[j].node = root;
            LeafNodes[j].env = env.Clone();

            Mutex mutex = new Mutex();
            SelectLeaf(ref LeafNodes[j].node, ref LeafNodes[j].env,ref mutex);
            mutex.ReleaseMutex();
        }
        Task<float>[] tasks = new Task<float>[_threads];

        for (int j = 0; j < _threads; j++)
        {
            int id = j;
            tasks[j] = Task.Run(() => RollOut(LeafNodes[id].node, LeafNodes[id].env));
        }
        int n = 0;

        while (tasks.Count() > 0)
        {
            int j = Task.WaitAny(tasks);

            if (LeafNodes[j].node.VisitCount >= 20 && LeafNodes[j].node.IsLeaf() && LeafNodes[j].env.IsEnd().Item2 == 2)
            {
                ExpandLeafNode(LeafNodes[j].node, LeafNodes[j].env);
            }

            LeafNodes[j].node.UpdateRecursive(tasks[j].Result);
            LeafNodes[j].node = root;
            LeafNodes[j].env = env.Clone();

            Mutex mutex2 = new Mutex(); 
            SelectLeaf(ref LeafNodes[j].node, ref LeafNodes[j].env,ref mutex2);
            mutex2.ReleaseMutex();

            tasks[j] = Task.Run(() => RollOut(LeafNodes[j].node, LeafNodes[j].env));

            if (n > RollOutTimes * _threads)
            {
                break;
            }
            n++;
        }
        /*
        for (int j = 0; j < _threads; j++)
        {
            if (LeafNodes[j].node.VisitCount >= 20 && LeafNodes[j].node.IsLeaf() && LeafNodes[j].env.IsEnd().Item2 == 2)
            {
                ExpandLeafNode(LeafNodes[j].node, LeafNodes[j].env);
            }
            LeafNodes[j].node.UpdateRecursive(tasks[j].Result);
        }
        */

        float[,] ActionProbsArray = new float[Global.SIZE, Global.SIZE];
        for (int i = 0; i < Global.SIZE; i++)
        {
            for (int j = 0; j < Global.SIZE; j++)
            {
                ActionProbsArray[i, j] = (float)root.Children[i, j].VisitCount / (float)root.VisitCount;
            }
        }

        return torch.tensor(ActionProbsArray);
    }
}