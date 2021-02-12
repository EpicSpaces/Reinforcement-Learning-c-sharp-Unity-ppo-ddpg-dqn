using UnityEngine;

public class AgentDDPG : MonoBehaviour
{
    ActorNet_DDPG anet;
    public CriticNet_DDPG cnet;
    ActorNet_DDPG atgt_net;
    CriticNet_DDPG ctgt_net;

    int batch_capacity;
    int batch_size;
    int updater ;
    int input_size ;
    int hidden_size;
    int output_size;
    float learning_rate;
    float max_grad_norm;
    float gamma;
    float epsilon;
    float decay;
    float min_decay;
    float b;

    float[,] bs;
    float[,] ba;
    float[] br;
    float[,] bs1;

    int bi = 0;
    int p = 0;
    public AgentDDPG(int batch_capacity,
    int batch_size,
    int updater,
    int input_size,
    int hidden_size,
    int output_size,
    float learning_rate,
    float max_grad_norm,
    float gamma,
    float epsilon,
    float decay,
    float min_decay,
    float b)
    {
        anet = new ActorNet_DDPG(input_size, hidden_size, output_size, learning_rate);
        atgt_net = new ActorNet_DDPG(input_size, hidden_size,  output_size,  learning_rate);
        cnet = new CriticNet_DDPG(input_size, output_size,  hidden_size,  learning_rate);
        ctgt_net = new CriticNet_DDPG(input_size, output_size,  hidden_size,  learning_rate);
        this.batch_capacity = batch_capacity;
        this.batch_size = batch_size;
        this.updater = updater;
        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.output_size = output_size;
        this.learning_rate = learning_rate;
        this.max_grad_norm = max_grad_norm;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.decay = decay;
        this.min_decay = min_decay;
        this.b = b;
        bs = new float[ batch_capacity,  input_size];
        ba = new float[ batch_capacity,  output_size];
        br = new float[ batch_capacity];
        bs1 = new float[ batch_capacity,  input_size];
    }
    public float[,] Predict(float[,] s)
    {
       float[,]a= anet.Forward(s,  b);
        float[,] a_noise=new float[a.GetLength(0), a.GetLength(1)];
        for (int i = 0; i < a.GetLength(0); i++)
        {
            for (int j = 0; j < a.GetLength(1); j++)
            {
                a_noise[0, j] = a[i, j] +  epsilon * Mathf.Sqrt(-2.0f * Mathf.Log(Random.Range(0.0f, 1.0f))) * Mathf.Sin(2.0f * Mathf.PI * Random.Range(0.0f, 1.0f));
                if (a_noise[i, 0] < - b)
                    a_noise[i, 0] = - b;
                else if (a_noise[i, 0] >  b)
                    a_noise[i, 0] =  b;
            }
        }
        return a_noise;
    }
    public float Train(float[,] s, float[,] a, float r)
    {
        float loss = 0;
        if (bi !=  batch_capacity)
        {
            for (int j = 0; j <  input_size; j++)
            {
                bs[bi, j] = s[0, j];
            }
            for (int j = 0; j <  output_size; j++)
            {
                ba[bi, j] = a[0, j];
            }
            br[bi] = r;
        }
        for (int i = 0; i <  input_size; i++)
        {
            int ns = bi - 1;
            if (ns <= 0)
            {
                ns = 0;
            }
            bs1[ns, i] = s[0, i];
        }

        if (bi ==  batch_capacity)
        {
            bi = -1;
            p = 1;
        }
        if (p == 1)
        {
            int n =  batch_capacity;
            int k =  batch_size;

            int[] pool = new int[n];
            int[] result = new int[k];
            for (int i = 0; i < n; i++)
            {
                pool[i] = i;
            }

            for (int i = 0; i < k; i++)
            {
                int j = Random.Range(0, n - i - 1);
                result[i] = pool[j];
                pool[j] = pool[n - i - 1];
            }

            float[,] ss = new float[k,  input_size];
            float[,] sa = new float[k,  output_size];
            float[] sr = new float[k];
            float[,] ss1 = new float[k,  input_size];
            
            for (int i = 0; i < k; i++)
            {
                sr[i] = br[result[i]];
                
                for (int j = 0; j <  input_size; j++)
                {
                    ss[i, j] = bs[result[i], j];
                    ss1[i, j] = bs1[result[i], j];
                }
                for (int j = 0; j <  output_size; j++)
                {
                    sa[i,j] = ba[result[i],j];
                }
            }
            float[,] target_q = ctgt_net.Forward(ss1,atgt_net.Forward(ss1, b));
            
            for (int i = 0; i < k; i++)
            {
                target_q[i,0] = sr[i] +  gamma * target_q[i, 0];
            }
            
            cnet.Backward(ss,sa,target_q, max_grad_norm);
            float[,] a_grads= cnet.Evaluate_action_grads(ss, anet.Forward(ss,  b));
            anet.Backward(ss,a_grads, b, max_grad_norm);
            if (bi %  updater == 0)
            {
                ctgt_net = new CriticNet_DDPG(cnet);
            }
            if (bi % ( updater+1) == 0)
            {
                atgt_net = new ActorNet_DDPG(anet);
            }

             epsilon = Mathf.Max( epsilon *  decay,  min_decay);
        }
        bi++;
        return loss;
    }
    int[] maxi(float[,] q)
    {
        int[] maxi = new int[q.GetLength(0)];
        float maxq = 0;
        for (int i = 0; i < q.GetLength(0); i++)
        {
            maxq = q[i, 0];
            for (int j = 0; j < q.GetLength(1); j++)
            {
                if (maxq < q[i, j])
                {
                    maxq = q[i, j];
                    maxi[i] = j;
                }
            }
        }
        return maxi;
    }
}
