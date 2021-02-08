using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentDQN : MonoBehaviour
{
    Net_DQN net;
    Net_DQN target_net;
    float[,] bs;
    int[] ba_idx;
    float[] br;
    float[,] bs1;

    int batch_capacity;
    int batch_size;
    int updater;
    int input_size;
    int hidden_size;
    int output_size;
    float learning_rate;
    float max_grad_norm;
    float gamma;
    float epsilon;
    float decay;
    float min_decay;
    int bi;
    int p;
    public AgentDQN(int batch_capacity,
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
    float min_decay)
    
    {
        net = new Net_DQN(input_size, hidden_size, output_size, learning_rate);
        target_net = new Net_DQN(input_size, hidden_size, output_size, learning_rate);

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

        bs = new float[batch_capacity, input_size];
        ba_idx = new int[batch_capacity];
        br = new float[ batch_capacity];
        bs1 = new float[ batch_capacity,  input_size];
    }
    public int Predict(float[,] s)
    {
        int act_idx = 0;
        if (Random.Range(0f, 1f) <  epsilon)
        {
            act_idx = Random.Range(0,  output_size);
        }
        else
        {
            float[,] q = net.Forward(s);
            int[] a_idx = maxi(q);
            act_idx = a_idx[0];
        }/*
        float[,] q = net.Forward(s);
        float[] softmax = new float[output_size];

        float sumexp = 0;
        for (int i = 0; i < output_size; i++)
        {
            sumexp += Mathf.Exp(q[0, i]);
        }
        for (int i = 0; i < output_size; i++)
        {
            softmax[i] = Mathf.Exp(q[0, i]) / sumexp;
        }

        float[] weightsum = new float[output_size];
        
        weightsum[0] = softmax[0];

        for (int i = 1; i < output_size; i++)
        {
            weightsum[i] = weightsum[i - 1] + softmax[i];
        }
        float kk = Random.Range(0, weightsum[weightsum.Length - 1]);
        int ii = 0;
        for (ii = 0; kk > weightsum[ii]; ii++) ;
            act_idx = ii;*/
        return act_idx;
    }
    public float Train(float[,] s, int act_idx, float r)
    {
        float loss = 0;
        if (bi !=  batch_capacity)
        {
            for (int j = 0; j <  input_size; j++)
            {
                bs[bi, j] = s[0, j];
            }
            ba_idx[bi] = act_idx;
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
            int[] sa_idx = new int[k];
            float[] sr = new float[k];
            float[,] ss1 = new float[k,  input_size];
            
            for (int i = 0; i < k; i++)
            {
                sr[i] = br[result[i]];
                sa_idx[i] = ba_idx[result[i]];
                for (int j = 0; j <  input_size; j++)
                {
                    ss[i, j] = bs[result[i], j];
                    ss1[i, j] = bs1[result[i], j];
                }
            }
            float[,] q1 = net.Forward(ss1);
            int[] max_q1 = maxi(q1);

            float[,] q2 = target_net.Forward(ss1);

            float[] q_target = new float[k];
            for (int j = 0; j < k; j++)
            {
                q_target[j] = sr[j] +  gamma * q2[j, max_q1[j]];
            }
            loss=net.Backward(ss, sa_idx, q_target,  max_grad_norm);

            if (bi %  updater == 0)
            {
                target_net = new Net_DQN(net);
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
