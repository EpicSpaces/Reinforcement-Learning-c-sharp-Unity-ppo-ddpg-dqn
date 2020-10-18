using UnityEngine;
using UnityEngine.UI;

public class Agent3 : MonoBehaviour
{
    AgentDQN adqn;
    DQN_Config dqn_config;
    public int batch_capacity = 2000;
    public int batch_size = 32;
    public int updater = 90;
    public int input_size = 3;
    public int hidden_size = 100;
    public int output_size = 80;
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

        for (int i = 0; i < output_size/2; i++)
        {
            action_list[i] =-i;
        }
        for (int i = output_size/2; i < output_size; i++)
        {
            action_list[i] = i-output_size/2;
        }
    }
    float losslabel=0;
    // Update is called once per frame
    void Update()
    {
        float[,] s = new float[1, input_size];
        s[0, 0] = transform.rotation.x;
        s[0, 1] = GetComponent<Rigidbody>().angularVelocity.x;
        s[0, 2] = transform.rotation.x;

        int act_idx = adqn.Predict(s);
        if (Input.GetKey(KeyCode.LeftArrow))
        {
           act_idx=15;
        }
        if (Input.GetKey(KeyCode.RightArrow))
        {
            act_idx=64;
        }
        GetComponent<Rigidbody>().AddForce( 0, 0, action_list[act_idx]);
        
        float r = Vector3.Angle(Vector3.up, -transform.up)- GetComponent<Rigidbody>().angularVelocity.magnitude;
        GameObject.Find("Canvas").transform.Find("Text").GetComponent<Text>().text = r + "_" + losslabel;
        float loss = adqn.Train(s, act_idx, r);
        if (loss != 0)
            losslabel = loss;
    GetComponent<Rigidbody>().angularVelocity *=0.999f;
       
    }
}
