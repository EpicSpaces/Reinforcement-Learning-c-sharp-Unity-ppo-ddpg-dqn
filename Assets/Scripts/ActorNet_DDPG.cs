using UnityEngine;
using static Matr.Matrix;
    class ActorNet_DDPG
{
        float[,] w1;
        float[,] w2;
        float[,] b1;
        float[,] b2;

        float[,] m1;
        float[,] v1;
        float[,] m2;
        float[,] v2;
    
        float[,] mb1;
        float[,] vb1;
        float[,] mb2;
        float[,] vb2;


        int t = 0;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float lr = 1e-3f;

    public ActorNet_DDPG(ActorNet_DDPG anet)
    {
        w1 = Init(anet.w1.GetLength(0), anet.w1.GetLength(1));
        w2 = Init(anet.w2.GetLength(0), anet.w2.GetLength(1));
        b1 = new float[1, anet.b1.GetLength(1)];
        b2 = new float[1, anet.b2.GetLength(1)];

        m1 = new float[anet.m1.GetLength(0), anet.m1.GetLength(1)];
        v1 = new float[anet.v1.GetLength(0), anet.v1.GetLength(1)];
        m2 = new float[anet.m2.GetLength(0), anet.m2.GetLength(1)];
        v2 = new float[anet.v2.GetLength(0), anet.v2.GetLength(1)];

        mb1 = new float[1, anet.mb1.GetLength(1)];
        vb1 = new float[1, anet.vb1.GetLength(1)];
        mb2 = new float[1, anet.mb2.GetLength(1)];
        vb2 = new float[1, anet.vb2.GetLength(1)];

        this.t = anet.t;
        this.lr = anet.lr;
        for (int j = 0; j < w1.GetLength(1); j++)
        {
            this.b1[0, j] = anet.b1[0, j];
            this.mb1[0, j] = anet.mb1[0, j];
            this.vb1[0, j] = anet.vb1[0, j];
            for (int i = 0; i < w1.GetLength(0); i++)
            {
                this.w1[i, j] = anet.w1[i, j];
                this.m1[i, j] = anet.m1[i, j];
                this.v1[i, j] = anet.v1[i, j];
            }
        }

        for (int j = 0; j < w2.GetLength(1); j++)
        {
            this.b2[0, j] = anet.b2[0, j];
            this.mb2[0, j] = anet.mb2[0, j];
            this.vb2[0, j] = anet.vb2[0, j];
            for (int i = 0; i < w2.GetLength(0); i++)
            {
                this.w2[i, j] = anet.w2[i, j];
                this.m2[i, j] = anet.m2[i, j];
                this.v2[i, j] = anet.v2[i, j];
            }
        }
    }
    public ActorNet_DDPG(int inp, int h, int outp, float lr)
        {
            w1 = Init(inp, h);
            w2 = Init(h, outp);
            b1 = new float[1, h];
            b2 = new float[1, outp];

            m1 = new float[inp, h];
            v1 = new float[inp, h];
            m2 = new float[h, outp];
            v2 = new float[h, outp];

            mb1 = new float[1, h];
            vb1 = new float[1, h];
            mb2 = new float[1, outp];
            vb2 = new float[1, outp];

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
        public float[,] Forward(float[,] s, float b)
        {
        float[,] z1 = Add(Dot(s, w1), b1);
        float[,] h1 = Relu(z1);
        float[,] z2 = Add(Dot(h1, w2), b2);
        
        float[,] a = new float[z2.GetLength(0),z2.GetLength(1)];
        for (int i = 0; i < z2.GetLength(0); i++)
        {
            for (int j = 0; j < z2.GetLength(1); j++)
            {
                z2[i, j] = Mathf.Clamp(z2[i, j], -20, 20);
                a[i,j] = b * (Mathf.Exp(2 * z2[i, j]) - 1) / (Mathf.Exp(2 * z2[i, j]) + 1);
            }
        }

        return a;
    }

    public void Backward(float[,] s, float[,] d,float b, float max_grad_norm)
    {
        float[,] z1 = Add(Dot(s, w1), b1);
        float[,] h1 = Relu(z1);
        float[,] z2 = Add(Dot(h1, w2), b2);
        float[,] grads = new float[z2.GetLength(0), z2.GetLength(1)];
        for (int i = 0; i < z2.GetLength(0); i++)
        {
            for (int j = 0; j < z2.GetLength(1); j++)
            {
                z2[i, j] = Mathf.Clamp(z2[i,j], -20,20);
                grads[i, j] = -d[i,j]*b * (1-(((Mathf.Exp(2 * z2[i, j]) - 1) / (Mathf.Exp(2 * z2[i, j]) + 1))* ((Mathf.Exp(2 * z2[i, j]) - 1) / (Mathf.Exp(2 * z2[i, j]) + 1))));
            }
        }
        int batch_size = s.GetLength(0);

        float[,] out1 = Dot(grads, Transpose(w2));
        out1 = DerRelu(out1, z1);

        float[,] dw2 = Dot(Transpose(h1), grads);
        float[,] dw1 = Dot(Transpose(s), out1);
        float[,] db2 = Sum(grads);
        float[,] db1 = Sum(out1);

        float total_norm = 0;

        total_norm += Sum_total_norm(w2, b2);
        total_norm += Sum_total_norm(w1, b1);

        total_norm = Mathf.Sqrt(total_norm);
        float clip_coef = (float)(max_grad_norm / (total_norm + 1e-6));
        if (clip_coef < 1)
        {
            dw2 = Mult_clip_coef(dw2, db2, out db2, clip_coef);
            dw1 = Mult_clip_coef(dw1, db1, out db1, clip_coef);

        }
        t++;
        Adam(w1, dw1, m1, v1, batch_size, beta1, beta2, eps, lr, t);
        Adam(w2, dw2, m2, v2, batch_size, beta1, beta2, eps, lr, t);
        Adam(b1, db1, mb1, vb1, batch_size, beta1, beta2, eps, lr, t);
        Adam(b2, db2, mb2, vb2, batch_size, beta1, beta2, eps, lr, t);
    }   
}



