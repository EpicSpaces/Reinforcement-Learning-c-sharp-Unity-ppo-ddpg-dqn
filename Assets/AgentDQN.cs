using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentDQN : MonoBehaviour
{
    Net net;
    Net target_net;
    public DQN_Config dqn_config;
    float[,] bs;
    int[] ba_idx;
    float[] br;
    float[,] bs1;

    int bi = 0;
    int p = 0;
    public AgentDQN(DQN_Config dqn_config)
    {
        net = new Net(dqn_config.input_size, dqn_config.hidden_size, dqn_config.output_size, dqn_config.learning_rate);
        target_net = new Net(dqn_config.input_size, dqn_config.hidden_size, dqn_config.output_size, dqn_config.learning_rate);
        
        this.dqn_config = new DQN_Config(dqn_config);

        bs = new float[dqn_config.batch_capacity, dqn_config.input_size];
        ba_idx = new int[dqn_config.batch_capacity];
        br = new float[dqn_config.batch_capacity];
        bs1 = new float[dqn_config.batch_capacity, dqn_config.input_size];
    }
    public int Predict(float[,] s)
    {
        int act_idx = 0;
        if (Random.Range(0f, 1f) < dqn_config.epsilon)
        {
            act_idx = Random.Range(0, dqn_config.output_size);
        }
        else
        {
            float[,] q = net.Forward(s);
            int[] a_idx = maxi(q);
            act_idx = a_idx[0];
        }
        return act_idx;
    }
    public float Train(float[,] s, int act_idx, float r)
    {
        float loss = 0;
        if (bi != dqn_config.batch_capacity)
        {
            for (int j = 0; j < dqn_config.input_size; j++)
            {
                bs[bi, j] = s[0, j];
            }
            ba_idx[bi] = act_idx;
            br[bi] = r;
        }
        for (int i = 0; i < dqn_config.input_size; i++)
        {
            int ns = bi - 1;
            if (ns <= 0)
            {
                ns = 0;
            }
            bs1[ns, i] = s[0, i];
        }

        if (bi == dqn_config.batch_capacity)
        {
            bi = -1;
            p = 1;
        }
        if (p == 1)
        {
            int n = dqn_config.batch_capacity;
            int k = dqn_config.batch_size;

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

            float[,] ss = new float[k, dqn_config.input_size];
            int[] sa_idx = new int[k];
            float[] sr = new float[k];
            float[,] ss1 = new float[k, dqn_config.input_size];
            
            for (int i = 0; i < k; i++)
            {
                sr[i] = br[result[i]];
                sa_idx[i] = ba_idx[result[i]];
                for (int j = 0; j < dqn_config.input_size; j++)
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
                q_target[j] = sr[j] + dqn_config.gamma * q2[j, max_q1[j]];
            }
            loss=net.Backward(ss, sa_idx, q_target, dqn_config.max_grad_norm);

            if (bi % dqn_config.updater == 0)
            {
                target_net = new Net(net);
            }

            dqn_config.epsilon = Mathf.Max(dqn_config.epsilon * dqn_config.decay, dqn_config.min_decay);
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
