using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentDDPG : MonoBehaviour
{
    ActorNet2 anet;
    CriticNet2 cnet;
    ActorNet2 atgt_net;
    CriticNet2 ctgt_net;

    public DDPG_Config ddpg_config;
    float[,] bs;
    float[,] ba;
    float[] br;
    float[,] bs1;

    int bi = 0;
    int p = 0;
    public AgentDDPG(DDPG_Config ddpg_config)
    {
        anet = new ActorNet2(ddpg_config.input_size, ddpg_config.hidden_size, ddpg_config.output_size, ddpg_config.learning_rate);
        atgt_net = new ActorNet2(ddpg_config.input_size, ddpg_config.hidden_size, ddpg_config.output_size, ddpg_config.learning_rate);
        cnet = new CriticNet2(ddpg_config.input_size, ddpg_config.output_size, ddpg_config.hidden_size, ddpg_config.learning_rate);
        ctgt_net = new CriticNet2(ddpg_config.input_size, ddpg_config.output_size, ddpg_config.hidden_size, ddpg_config.learning_rate);
        
        this.ddpg_config = new DDPG_Config(ddpg_config);

        bs = new float[ddpg_config.batch_capacity, ddpg_config.input_size];
        ba = new float[ddpg_config.batch_capacity, ddpg_config.output_size];
        br = new float[ddpg_config.batch_capacity];
        bs1 = new float[ddpg_config.batch_capacity, ddpg_config.input_size];
    }
    public float[,] Predict(float[,] s)
    {
       float[,]a= anet.Forward(s, ddpg_config.b);
        float[,] a_noise=new float[a.GetLength(0), a.GetLength(1)];
        for (int i = 0; i < a.GetLength(0); i++)
        {
            for (int j = 0; j < a.GetLength(1); j++)
            {
                a_noise[0, j] = a[i, j] + ddpg_config.epsilon * Mathf.Sqrt(-2.0f * Mathf.Log(Random.Range(0.0f, 1.0f))) * Mathf.Sin(2.0f * Mathf.PI * Random.Range(0.0f, 1.0f));
                if (a_noise[i, 0] < -ddpg_config.b)
                    a_noise[i, 0] = -ddpg_config.b;
                else if (a_noise[i, 0] > ddpg_config.b)
                    a_noise[i, 0] = ddpg_config.b;
            }
        }
        return a_noise;
    }
    public float Train(float[,] s, float[,] a, float r)
    {
        float loss = 0;
        if (bi != ddpg_config.batch_capacity)
        {
            for (int j = 0; j < ddpg_config.input_size; j++)
            {
                bs[bi, j] = s[0, j];
            }
            for (int j = 0; j < ddpg_config.output_size; j++)
            {
                ba[bi, j] = a[0, j];
            }
            br[bi] = r;
        }
        for (int i = 0; i < ddpg_config.input_size; i++)
        {
            int ns = bi - 1;
            if (ns <= 0)
            {
                ns = 0;
            }
            bs1[ns, i] = s[0, i];
        }

        if (bi == ddpg_config.batch_capacity)
        {
            bi = -1;
            p = 1;
        }
        if (p == 1)
        {
            int n = ddpg_config.batch_capacity;
            int k = ddpg_config.batch_size;

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

            float[,] ss = new float[k, ddpg_config.input_size];
            float[,] sa = new float[k, ddpg_config.output_size];
            float[] sr = new float[k];
            float[,] ss1 = new float[k, ddpg_config.input_size];
            
            for (int i = 0; i < k; i++)
            {
                sr[i] = br[result[i]];
                
                for (int j = 0; j < ddpg_config.input_size; j++)
                {
                    ss[i, j] = bs[result[i], j];
                    ss1[i, j] = bs1[result[i], j];
                }
                for (int j = 0; j < ddpg_config.output_size; j++)
                {
                    sa[i,j] = ba[result[i],j];
                }
            }
            float[,] target_q = ctgt_net.Forward(ss1,atgt_net.Forward(ss1,ddpg_config.b));
            
            for (int i = 0; i < k; i++)
            {
                target_q[i,0] = sr[i] + ddpg_config.gamma * target_q[i, 0];
            }
            
            cnet.Backward(ss,sa,target_q,ddpg_config.max_grad_norm);
            float[,] a_grads= cnet.Evaluate_action_grads(ss, anet.Forward(ss, ddpg_config.b));
            anet.Backward(ss,a_grads,ddpg_config.b,ddpg_config.max_grad_norm);
            if (bi % ddpg_config.updater == 0)
            {
                ctgt_net = new CriticNet2(cnet);
            }
            if (bi % (ddpg_config.updater+1) == 0)
            {
                atgt_net = new ActorNet2(anet);
            }

            ddpg_config.epsilon = Mathf.Max(ddpg_config.epsilon * ddpg_config.decay, ddpg_config.min_decay);
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
