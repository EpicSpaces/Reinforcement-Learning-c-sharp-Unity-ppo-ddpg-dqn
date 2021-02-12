using UnityEngine;
namespace Matr
{
    public static class Matrix
    {
        public static void Adam(  float[,] x,   float[,] dx,   float[,] m,   float[,] v,   int batch_size,   float beta1,   float beta2,   float eps,   float lr,  int t)
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
        public static float[,] Dot(  float[,] a,   float[,] b)
        {
            float[,] c = new float[a.GetLength(0), b.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int k = 0; k < b.GetLength(1); k++)
                {
                    for (int j = 0; j < b.GetLength(0); j++)
                    {
                        c[i, k] += a[i, j] * b[j, k];
                    }
                }
            }
            return c;
        }

        public static float[,] Transpose(  float[,] a)
        {
            float[,] b = new float[a.GetLength(1), a.GetLength(0)];
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    b[j, i] = a[i, j];
                }
            }
            return b;
        }
        public static float[,] Add(  float[,] a,   float[,] b)
        {
            float[,] c = new float[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    c[i, j] = a[i, j] + b[0, j];
                }
            }
            return c;
        }
        public static float[,] Relu(  float[,] a)
        {
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    if (a[i, j] < 0)
                        a[i, j] = 0;
                }
            }
            return a;
        }
        public static float[,] DerRelu(  float[,] a,   float[,] z)
        {
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    if (z[i, j] <= 0)
                        a[i, j] = 0;
                }
            }
            return a;
        }
        public static float[,] Sum(  float[,] a)
        {
            float[,] b = new float[1, a.GetLength(1)];
            for (int j = 0; j < a.GetLength(1); j++)
            {
                for (int i = 0; i < a.GetLength(0); i++)
                {
                    b[0, j] += a[i, j];
                }
            }
            return b;
        }
        public static float Sum_total_norm(  float[,] w,   float[,] b)
        {
            float total_norm = 0;
            for (int j = 0; j < w.GetLength(1); j++)
            {
                total_norm += b[0, j] * b[0, j];
                for (int i = 0; i < w.GetLength(0); i++)
                {
                    total_norm += w[i, j] * w[i, j];
                }
            }
            return total_norm;
        }
        public static float[,] Mult_clip_coef(  float[,] dw,   float[,] db, out float[,] rdb,   float clip_coef)
        {
            for (int j = 0; j < dw.GetLength(1); j++)
            {
                db[0, j] *= clip_coef;
                for (int i = 0; i < dw.GetLength(0); i++)
                {
                    dw[i, j] *= clip_coef;
                }
            }
            rdb = db;
            return dw;
        }
    }
}