using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

public class Agent4 : MonoBehaviour
{
    AgentDQN adqn;
    DQN_Config dqn_config;
    public int batch_capacity = 2000;
    public int batch_size = 32;
    public int updater = 90;
    public int input_size = 20;
    public int hidden_size = 100;
    public int output_size = 6;
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
        2*input_size,
        hidden_size,
        output_size,
        learning_rate,
        max_grad_norm,
        gamma,
        epsilon,
        decay,
        min_decay);
        adqn = new AgentDQN(dqn_config);
        /*action_list = new float[2*output_size];

        for (int i = -output_size; i < output_size; i++)
        {
            action_list[i+output_size] =i;
        }
        action_list[output_size] = output_size;*/
    }
    float losslabel=0;
    // Update is called once per frame
    private void FixedUpdate()
    {
        int layerMask = 1 << 8;
        //layerMask = ~layerMask;
        float[,] s = new float[1, 2 * 20];
        float r=0;
        r = -0.6f;
        for (int i = -20; i < 20; i++)
        {
            Vector3 v = transform.TransformDirection(Vector3.forward);
            Vector3 rv = Quaternion.AngleAxis(i*4, Vector3.up)*v;
            Debug.DrawRay(transform.position, 10 * rv, Color.green);
        
            RaycastHit hit;
            if (Physics.Raycast(transform.position, rv, out hit, Mathf.Infinity, layerMask))
            {
                s[0, (i+20)*1] = hit.distance;
                if (hit.collider.name == "Banana"&&hit.distance<3f) 
                {
                    //r += 0.5f; 
                      r += 0.0f; 
                }
                if (hit.collider.name == "Banana" && i==0)
                {
                    r += 0.4f;
                }

                Debug.DrawRay(transform.position, rv * hit.distance, Color.yellow);
            }
            else
            {
                Debug.DrawRay(transform.position, rv * 1000, Color.white);
            }
        }
        int act_idx = adqn.Predict(s);
        if (Input.GetKey(KeyCode.LeftArrow))
        {
            act_idx = 15;
        }
        if (Input.GetKey(KeyCode.RightArrow))
        {
            act_idx = 64;
        }
     
        r/=40;
        if (act_idx == 0)
            transform.Translate(new Vector3(1, 0, 0));
        if (act_idx==1)
        transform.Translate(new Vector3(1,0,0));
        if (act_idx == 2)
            transform.Translate(new Vector3(0, 0, 1));
        if (act_idx == 3)
            transform.Translate(new Vector3(-1, 0, 0));
        if (act_idx == 4)
            transform.Translate(new Vector3(0, 0, -1));
        if (act_idx == 5)
            transform.Rotate(new Vector3(0, -1, 0));
        if (act_idx == 6)
            transform.Rotate(new Vector3(0, 1, 0));
        
        float x = Mathf.Clamp(transform.position.x, -5, 5);
        float y = Mathf.Clamp(transform.position.y, -5, 5);
        float z = Mathf.Clamp(transform.position.z, -5, 5);
        transform.position = new Vector3(x, y, z);

        GameObject.Find("Canvas").transform.Find("Text").GetComponent<Text>().text = r + "_"+losslabel;
        float loss=adqn.Train(s, act_idx, r);
        if (loss != 0)
            losslabel = loss*loss;
        }
}
