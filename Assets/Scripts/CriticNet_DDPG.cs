using UnityEngine;
using static Matr.Matrix;
    public class CriticNet_DDPG
{
        float[,] w1;
        float[,] w2;
        float[,] w3;
        float[,] b1;
        float[,] b3;

        float[,] m1;
        float[,] v1;
        float[,] m2;
        float[,] v2;
    float[,] m3;
        float[,] v3;
    
        float[,] mb1;
        float[,] vb1;
        float[,] mb3;
        float[,] vb3;


        int t = 0;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float lr = 1e-3f;
    public CriticNet_DDPG(CriticNet_DDPG cnet)
    {
        w1 = Init(cnet.w1.GetLength(0), cnet.w1.GetLength(1));
        w2 = Init(cnet.w2.GetLength(0), cnet.w2.GetLength(1));
        w3 = Init(cnet.w3.GetLength(0), cnet.w3.GetLength(1));
        b1 = new float[1, cnet.b1.GetLength(1)];
        b3 = new float[1, cnet.b3.GetLength(1)];

        m1 = new float[cnet.m1.GetLength(0), cnet.m1.GetLength(1)];
        v1 = new float[cnet.v1.GetLength(0), cnet.v1.GetLength(1)];
        m2 = new float[cnet.m2.GetLength(0), cnet.m2.GetLength(1)];
        v2 = new float[cnet.v2.GetLength(0), cnet.v2.GetLength(1)];
        m3 = new float[cnet.m3.GetLength(0), cnet.m3.GetLength(1)];
        v3 = new float[cnet.v3.GetLength(0), cnet.v3.GetLength(1)];

        mb1 = new float[1, cnet.mb1.GetLength(1)];
        vb1 = new float[1, cnet.vb1.GetLength(1)];
        mb3 = new float[1, cnet.mb3.GetLength(1)];
        vb3 = new float[1, cnet.vb3.GetLength(1)];

        this.t = cnet.t;
        this.lr = cnet.lr;
        for (int j = 0; j < w1.GetLength(1); j++)
        {
            this.b1[0, j] = cnet.b1[0, j];
            this.mb1[0, j] = cnet.mb1[0, j];
            this.vb1[0, j] = cnet.vb1[0, j];
            for (int i = 0; i < w1.GetLength(0); i++)
            {
                this.w1[i, j] = cnet.w1[i, j];
                this.m1[i, j] = cnet.m1[i, j];
                this.v1[i, j] = cnet.v1[i, j];
            }
        }

        for (int j = 0; j < w2.GetLength(1); j++)
        {
            for (int i = 0; i < w2.GetLength(0); i++)
            {
                this.w2[i, j] = cnet.w2[i, j];
                this.m2[i, j] = cnet.m2[i, j];
                this.v2[i, j] = cnet.v2[i, j];
            }
        }
        for (int j = 0; j < w3.GetLength(1); j++)
        {
            this.b3[0, j] = cnet.b3[0, j];
            this.mb3[0, j] = cnet.mb3[0, j];
            this.vb3[0, j] = cnet.vb3[0, j];
            for (int i = 0; i < w3.GetLength(0); i++)
            {
                this.w3[i, j] = cnet.w3[i, j];
                this.m3[i, j] = cnet.m3[i, j];
                this.v3[i, j] = cnet.v3[i, j];
            }
        }
    }
    public CriticNet_DDPG(int inp, int inp2, int h, float lr)
        {
            w1 = Init(inp, h);
            w2 = Init(inp2, h);
            w3 = Init(h, 1);
            b1 = new float[1, h];
            b3 = new float[1, 1];

            m1 = new float[inp, h];
            v1 = new float[inp, h];
            m2 = new float[inp2, h];
            v2 = new float[inp2, h];
            m3 = new float[h, 1];
            v3 = new float[h, 1];

            mb1 = new float[1, h];
            vb1 = new float[1, h];
            mb3 = new float[1, 1];
            vb3 = new float[1, 1];

        this.lr = lr;
        }
        public float[,] Init(int inp, int outp)
        {
            float[,] u = new float[inp, outp];
            float un = Mathf.Sqrt(1.0f / (inp * outp));

            for (int i = 0; i < inp; i++)
            {
                for (int j = 0; j < outp; j++)
                {
                    u[i, j] = Random.Range(-un,un);
                }
            }
            return u;
        }
        public float[,] Forward(float[,] s, float[,] a)
        {
        float[,] z1 = Dot(s, w1);
        z1 = Add(Add(z1, Dot(a, w2)),b1);
        float[,] h1 = Relu(z1);
        float[,] v = Add(Dot(h1, w3), b3);
      
        return v;
    }

    public void Backward(float[,] s, float[,] a, float[,] target_q,float max_grad_norm)
    {

        float[,] z1 = Add(Dot(s, w1), b1);
        z1 = Add(Add(z1, Dot(a, w2)), b1);
        float[,] h1 = Relu(z1);
        float[,] v = Add(Dot(h1, w3), b3);

        float[,] grads = new float[v.GetLength(0), v.GetLength(1)];
        for (int i = 0; i < v.GetLength(0); i++)
        {
            grads[i, 0] = v[i, 0] - target_q[i, 0];
            if (grads[i, 0] < -1)
                grads[i, 0] = -1;
            else if (grads[i, 0] > 1)
                grads[i, 0] = 1;
        }
        
        int batch_size = s.GetLength(0);

        float[,] out1 = Dot(grads, Transpose(w3));
        out1 = DerRelu(out1, z1);

        float[,] dw3 = Dot(Transpose(h1), grads);
        float[,] dw2 = Dot(Transpose(a), out1);
        float[,] dw1 = Dot(Transpose(s), out1);
        float[,] db3 = Sum(grads);
        float[,] db1 = Sum(out1);

        float total_norm = 0;
        float[,] nothing = new float[1, w2.GetLength(1)];
        total_norm += Sum_total_norm(w3, b3);
        total_norm += Sum_total_norm(w2, nothing);
        total_norm += Sum_total_norm(w1, b1);

        total_norm = Mathf.Sqrt(total_norm);
        float clip_coef = (float)(max_grad_norm / (total_norm + 1e-6));
        if (clip_coef < 1)
        {
            dw3 = Mult_clip_coef(dw3, db3, out db3, clip_coef);
            dw2 = Mult_clip_coef(dw2, nothing, out nothing, clip_coef);
            dw1 = Mult_clip_coef(dw1, db1, out db1, clip_coef);
        }
        t++;
        Adam(w1, dw1, m1, v1, batch_size, beta1, beta2, eps, lr, t);
        Adam(w2, dw2, m2, v2, batch_size, beta1, beta2, eps, lr, t);
        Adam(w3, dw3, m3, v3, batch_size, beta1, beta2, eps, lr, t);
        Adam(b1, db1, mb1, vb1, batch_size, beta1, beta2, eps, lr, t);
        Adam(b3, db3, mb3, vb3, batch_size, beta1, beta2, eps, lr, t);
    }
    public float[,] Evaluate_action_grads(float[,] s, float[,] a)
    {

        float[,] z1 = Add(Dot(s, w1), b1);
        z1 = Add(Add(z1, Dot(a, w2)), b1);
        float[,] h1 = Relu(z1);
        float[,] v = Add(Dot(h1, w3), b3);

        float[,] grads = new float[v.GetLength(0), v.GetLength(1)];
        for (int i = 0; i < v.GetLength(0); i++)
        {
            for (int j = 0; j < v.GetLength(1); j++)
            {
                grads[i, j] = 1;
            }
        }

        int batch_size = s.GetLength(0);

        float[,] out1 = Dot(grads, Transpose(w3));
        out1 = DerRelu(out1, z1);
        return Dot(out1, Transpose(w2));
    }
    }



