using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentPPO : MonoBehaviour
{
    ActorNet anet;
    CriticNet cnet;
    int batch_Capacity;
    int batch_size;
    int ppo_epoch;
    int input_size;
    int hidden_size;
    int output_size;
    float learning_rate_a;
    float learning_rate_v;
    float gamma;
    float b;
    float clip_param;
    float max_grad_norm;

    float[,] bs;
    float[,] ba;
    float[] br;
    float[,] bs1;
    float[,] bolp;

    int bi = 0;

    public AgentPPO(int batch_Capacity,
    int batch_size,
    int ppo_epoch,
    int input_size,
    int hidden_size,
    int output_size,
    float learning_rate_a,
    float learning_rate_v,
    float gamma,
    float b,
    float clip_param,
    float max_grad_norm
    )
    {
        anet = new ActorNet(input_size, hidden_size, output_size, learning_rate_a);
        cnet = new CriticNet(input_size, hidden_size, output_size, learning_rate_v);

        bs = new float[batch_Capacity, input_size];
        ba = new float[batch_Capacity, output_size];
        br = new float[batch_Capacity];
        bs1 = new float[batch_Capacity, input_size];
        bolp = new float[batch_Capacity, output_size];

    }
    public float[,] Predict(float[,] s,out float[,] alp, float b)
    {
        alp = new float[1, output_size];
        float[,] a = anet.Forward(s, b, out alp);
        return a;
    }
    public float Train(float[,] s, float[,] a, float r, float[,] alp)
    {
        float loss = 0;
        if (bi != batch_Capacity)
        {
            for (int j = 0; j < input_size; j++)
            {
                bs[bi, j] = s[0, j];
            }
            for (int j = 0; j < output_size; j++)
            {
                ba[bi, j] = a[0, j];
                bolp[bi, j] = alp[0, j];
            }
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
                float[,] sa = new float[k, output_size];
                float[,] ss1 = new float[k, input_size];
                float[,] solp = new float[k, output_size];
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
                    for (int j = 0; j < output_size; j++)
                    {
                        sa[i, j] = ba[result[i], j];
                        solp[i, j] = bolp[result[i], j];
                    }
                }
                anet.Backward(ss, sa, solp, sadv, b, clip_param, max_grad_norm);
                loss=cnet.Backward(ss, starget_v, max_grad_norm);
            }
            bi = -1;
        }
        bi++;
        return loss;
    }
}
