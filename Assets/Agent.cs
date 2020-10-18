using UnityEngine;
using UnityEngine.UI;

public class Agent : MonoBehaviour
{
    AgentDQN adqn;
    DQN_Config dqn_config;
    public int batch_capacity = 2000;
    public int batch_size = 32;
    public int updater = 30;
    public int input_size = 3;
    public int hidden_size = 100;
    public int output_size = 5;
    public float learning_rate = 1e-3f;
    public float max_grad_norm = 0.5f;
    public float gamma = 0.99f;
    public float epsilon = 1f;
    public float decay = 0.999f;
    public float min_decay = 0.01f;
    public float[] action_list;
    void Start()
    {
        dqn_config = new DQN_Config(
        batch_capacity,
        batch_size,
        updater,
        input_size,
        hidden_size,
        output_size,
        learning_rate,
        max_grad_norm,
        gamma,
        epsilon,
        decay,
        min_decay);
        adqn = new AgentDQN(dqn_config);
        action_list = new float[output_size];
    }

    // Update is called once per frame
    void Update()
    {
        float[,] s = new float[1, input_size];
        s[0, 0] = transform.position.y;
        s[0, 1] = transform.position.z;
        s[0, 2] = transform.position.x;

        int act_idx = adqn.Predict(s);

        transform.Translate(action_list[act_idx], 0, 0);

        float r = -(transform.position - GameObject.Find("Cube (2)").transform.position).magnitude;

        GameObject.Find("Canvas").transform.Find("Text").GetComponent<Text>().text = r + "";
        adqn.Train(s, act_idx, r);
    }
}
