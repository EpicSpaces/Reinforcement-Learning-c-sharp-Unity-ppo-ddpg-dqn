using UnityEngine;
using UnityEngine.UI;

public class AgentP : MonoBehaviour
{
    AgentPPO agentppo;
    PPO_Config ppo_config;
    public int batch_Capacity = 1000;
    public int batch_size = 32;
    public int ppo_epoch = 10;
    public int input_size = 3;
    public int hidden_size = 100;
    public int output_size = 1;
    public float learning_rate_a = 1e-4f;
    public float learning_rate_v = 3e-4f;
    public float gamma = 0.99f;
    public float b = 2.0f;
    public float clip_param = 0.2f;
    public float max_grad_norm = 0.5f;
    void Start()
    {
        ppo_config = new PPO_Config(
        batch_Capacity,
        batch_size,
   ppo_epoch,
   input_size,
   hidden_size,
   output_size,
     learning_rate_a,
     learning_rate_v,
     gamma,
     b,
     clip_param,
     max_grad_norm);
        agentppo = new AgentPPO(ppo_config);
    }

    // Update is called once per frame
    void Update()
    {
        float[,] s = new float[1, input_size];
        s[0, 0] = transform.position.y;
        s[0, 1] = transform.position.z;
        s[0, 2] = transform.position.x;

        float[,] alp = new float[1, output_size];
        float[,] a = agentppo.Predict(s, out alp, b);

        transform.Translate(a[0, 0], 0, 0);

        float r = -(transform.position - GameObject.Find("Cube (2)").transform.position).magnitude;

        GameObject.Find("Canvas").transform.Find("Text").GetComponent<Text>().text = r + "";

        agentppo.Train(s,a,r, alp);
    }
}
