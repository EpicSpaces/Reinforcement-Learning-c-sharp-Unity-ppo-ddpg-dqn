using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentPPO : MonoBehaviour
{
    ActorNet anet;
    CriticNet cnet;
    PPO_Config ppo_config;

    float[,] bs;
    float[,] ba;
    float[] br;
    float[,] bs1;
    float[,] bolp;

    int bi = 0;

    public AgentPPO(PPO_Config ppo_config)
    {
        anet = new ActorNet(ppo_config.input_size, ppo_config.hidden_size, ppo_config.output_size, ppo_config.learning_rate_a);
        cnet = new CriticNet(ppo_config.input_size, ppo_config.hidden_size, ppo_config.output_size, ppo_config.learning_rate_v);

        this.ppo_config = new PPO_Config(ppo_config);

        bs = new float[ppo_config.batch_Capacity, ppo_config.input_size];
        ba = new float[ppo_config.batch_Capacity, ppo_config.output_size];
        br = new float[ppo_config.batch_Capacity];
        bs1 = new float[ppo_config.batch_Capacity, ppo_config.input_size];
        bolp = new float[ppo_config.batch_Capacity, ppo_config.output_size];

    }
    public float[,] Predict(float[,] s,out float[,] alp, float b)
    {
        alp = new float[1, ppo_config.output_size];
        float[,] a = anet.Forward(s, b, out alp);
        return a;
    }
    public float Train(float[,] s, float[,] a, float r, float[,] alp)
    {
        float loss = 0;
        if (bi != ppo_config.batch_Capacity)
        {
            for (int j = 0; j < ppo_config.input_size; j++)
            {
                bs[bi, j] = s[0, j];
            }
            for (int j = 0; j < ppo_config.output_size; j++)
            {
                ba[bi, j] = a[0, j];
                bolp[bi, j] = alp[0, j];
            }
            br[bi] = r;
        }
        for (int i = 0; i < ppo_config.input_size; i++)
        {
            int ns = bi - 1;
            if (ns <= 0)
            {
                ns = 0;
            }
            bs1[ns, i] = s[0, i];
        }

        if (bi == ppo_config.batch_Capacity)
        {
            int n = ppo_config.batch_Capacity;
            int k = ppo_config.batch_size;

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

                target_v[i] = br[i] + ppo_config.gamma * cnet.Forward(bs1)[i, 0];
                adv[i] = target_v[i] - cnet.Forward(bs)[i, 0];
            }

            for (int ii = 0; ii < ppo_config.ppo_epoch * ppo_config.batch_size; ii++)
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

                float[,] ss = new float[k, ppo_config.input_size];
                float[,] sa = new float[k, ppo_config.output_size];
                float[,] ss1 = new float[k, ppo_config.input_size];
                float[,] solp = new float[k, ppo_config.output_size];
                float[] starget_v = new float[k];
                float[] sadv = new float[k];

                for (int i = 0; i < k; i++)
                {
                    starget_v[i] = target_v[result[i]];
                    sadv[i] = adv[result[i]];
                    for (int j = 0; j < ppo_config.input_size; j++)
                    {
                        ss[i, j] = bs[result[i], j];
                    }
                    for (int j = 0; j < ppo_config.output_size; j++)
                    {
                        sa[i, j] = ba[result[i], j];
                        solp[i, j] = bolp[result[i], j];
                    }
                }
                anet.Backward(ss, sa, solp, sadv, ppo_config.b, ppo_config.clip_param, ppo_config.max_grad_norm);
                loss=cnet.Backward(ss, starget_v, ppo_config.max_grad_norm);
            }
            bi = -1;
        }
        bi++;
        return loss;
    }
}
