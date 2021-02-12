using UnityEngine;
using static Matr.Matrix;
class Net_DQN
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
    public Net_DQN(Net_DQN net)
        {
        w1 = Init(net.w1.GetLength(0), net.w1.GetLength(1));
        w2 = Init(net.w2.GetLength(0), net.w2.GetLength(1));
        b1 = new float[1, net.b1.GetLength(1)];
        b2 = new float[1, net.b2.GetLength(1)];

        m1 = new float[net.m1.GetLength(0), net.m1.GetLength(1)];
        v1 = new float[net.v1.GetLength(0), net.v1.GetLength(1)];
        m2 = new float[net.m2.GetLength(0), net.m2.GetLength(1)];
        v2 = new float[net.v2.GetLength(0), net.v2.GetLength(1)];

        mb1 = new float[1, net.mb1.GetLength(1)];
        vb1 = new float[1, net.vb1.GetLength(1)];
        mb2 = new float[1, net.mb2.GetLength(1)];
        vb2 = new float[1, net.vb2.GetLength(1)];

        this.t = net.t;
        this.lr = net.lr;
        for (int j = 0; j < w1.GetLength(1); j++)
        {
            this.b1[0, j] = net.b1[0, j];
            this.mb1[0, j] = net.mb1[0, j];
            this.vb1[0, j] = net.vb1[0, j];
            for (int i = 0; i < w1.GetLength(0); i++)
            {
                this.w1[i, j] = net.w1[i, j];
                this.m1[i, j] = net.m1[i, j];
                this.v1[i, j] = net.v1[i, j];
            }
        }

        for (int j = 0; j < w2.GetLength(1); j++)
        {
            this.b2[0, j] = net.b2[0, j];
            this.mb2[0, j] = net.mb2[0, j];
            this.vb2[0, j] = net.vb2[0, j];
            for (int i = 0; i < w2.GetLength(0); i++)
            {
                this.w2[i, j] = net.w2[i, j];
                this.m2[i, j] = net.m2[i, j];
                this.v2[i, j] = net.v2[i, j];
            }
        }
    }
        public Net_DQN(int inp, int h, int outp, float lr)
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

        public void Adam(float[,] x, float[,] dx, float[,] m, float[,] v, int batch_size)
        {
            for (int i = 0; i < x.GetLength(0); i++)
            {
                for (int j = 0; j < x.GetLength(1); j++)
                {
                    float d = dx[i, j] / batch_size;
                    m[i, j] = m[i, j] * beta1 + (1 - beta1) * d;
                    v[i, j] = v[i, j] * beta2 + (1 - beta2) * d * d;
                    float mb = m[i, j] / (1 - Mathf.Pow(beta1, t));
                    float vb = v[i, j] / (1 - Mathf.Pow(beta2, t));

                    x[i, j] = x[i, j] - lr * (mb / (Mathf.Sqrt(vb) + eps));
                }
            }

        }

        public float[,] Forward(float[,] s)
        {
            float[,] z1 = Add(Dot(s, w1), b1);
            float[,] h1 = Relu(z1);
            return Add(Dot(h1, w2), b2);
        }

        public float Backward(float[,] s, int[] a_idx, float[] q_target, float max_grad_norm)
        {
            float[,] z1 = Add(Dot(s, w1), b1);
            float[,] h1 = Relu(z1);
            float[,] q = Add(Dot(h1, w2), b2);

            float[,] d = new float[q.GetLength(0), q.GetLength(1)];
            for (int i = 0; i < s.GetLength(0); i++)
            {
                d[i, a_idx[i]] = q[i, a_idx[i]] - q_target[i];
                if (d[i, a_idx[i]] < -1)
                    d[i, a_idx[i]] = -1;
                else if (d[i, a_idx[i]] > 1)
                    d[i, a_idx[i]] = 1;
            }
            float[,] out1 = Dot(d, Transpose(w2));
            out1 = DerRelu(out1, z1);

            float[,] dw2 = Dot(Transpose(h1), d);
            float[,] dw1 = Dot(Transpose(s), out1);
            float[,] db2 = Sum(d);
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
            Adam(w1, dw1, m1, v1, s.GetLength(0));
            Adam(w2, dw2, m2, v2, s.GetLength(0));
            Adam(b1, db1, mb1, vb1, s.GetLength(0));
            Adam(b2, db2, mb2, vb2, s.GetLength(0));
       
        return Sum_total_norm(d, new float[1,d.GetLength(1)]) / s.GetLength(0);
    }
}



