using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentPPO_D : MonoBehaviour
{
    ActorNet_PPO_D anet;
    CriticNet_PPO_D cnet;

    int batch_Capacity;
    int batch_size;
    int ppo_epoch;
    int input_size;
    int hidden_size;
    int output_size;
    float learning_rate_a;
    float learning_rate_v;
    float gamma;
    float clip_param;
    float max_grad_norm;

    float[,] bs;
    int[] ba;
    float[] br;
    float[,] bs1;
    float[] bap;

    int bi = 0;

    public AgentPPO_D(int batch_Capacity,
    int batch_size,
    int ppo_epoch,
    int input_size,
    int hidden_size,
    int output_size,
    float learning_rate_a,
    float learning_rate_v,
    float gamma,
    float clip_param,
    float max_grad_norm)
    {
        anet = new ActorNet_PPO_D(input_size, hidden_size, output_size, learning_rate_a);
        cnet = new CriticNet_PPO_D(input_size, hidden_size, 1, learning_rate_v);

        bs = new float[batch_Capacity, input_size];
        ba = new int[batch_Capacity];
        br = new float[batch_Capacity];
        bs1 = new float[batch_Capacity, input_size];
        bap = new float[batch_Capacity];

    }
    public int Predict(float[,] s,out float ap)
    {
        ap = 0;
        int a_idx = anet.Forward(s, out ap);
        return a_idx;
    }
    public float Train(float[,] s, int a, float r, float ap)
    {
        float loss = 0;
        if (bi != batch_Capacity)
        {
            for (int j = 0; j < input_size; j++)
            {
                bs[bi, j] = s[0, j];
            }
            bap[bi] = ap;
            ba[bi] = a;
            br[bi] = r;
        }
        for (int i = 0; i < input_size; i++)
        {
            int ns = bi - 1;
            if (ns <= 0)
            {
                ns = 0;
            }
            bs1[ns, i] = s[0, i];
        }

        if (bi == batch_Capacity)
        {
            int n = batch_Capacity;
            int k = batch_size;

            float mean = 0;
            for (int i = 0; i < n; i++)
            {
                mean += br[i];
            }
            mean /= n;
            float sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += (br[i] - mean) * (br[i] - mean);
            }
            float std = Mathf.Sqrt(sum / (n - 1));

            float[] target_v = new float[n];
            float[] adv = new float[n];

            for (int i = 0; i < n; i++)
            {
                br[i] = (float)((br[i] - mean) / (std + 1e-5));

                target_v[i] = br[i] + gamma * cnet.Forward(bs1)[i, 0];
                adv[i] = target_v[i] - cnet.Forward(bs)[i, 0];
            }

            for (int ii = 0; ii < ppo_epoch * batch_size; ii++)
            {
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

                float[,] ss = new float[k, input_size];
                int[] sa_idx = new int[k];
                float[,] ss1 = new float[k, input_size];
                float[] sap = new float[k];
                float[] starget_v = new float[k];
                float[] sadv = new float[k];

                for (int i = 0; i < k; i++)
                {
                    starget_v[i] = target_v[result[i]];
                    sadv[i] = adv[result[i]];
                    for (int j = 0; j < input_size; j++)
                    {
                        ss[i, j] = bs[result[i], j];
                    }
                    sap[i] = bap[result[i]];
                    sa_idx[i] = ba[result[i]];
                }
                anet.Backward(ss, sa_idx, sap, sadv, clip_param, max_grad_norm);
                loss=cnet.Backward(ss, starget_v, max_grad_norm);
            }
            bi = -1;
        }
        bi++;
        return loss;
    }
}
