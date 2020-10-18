using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

public class Agent555ppo : MonoBehaviour
{
    AgentPPO agentppo;
    PPO_Config ppo_config;
    public int batch_Capacity = 1000;
    public int batch_size = 32;
    public int ppo_epoch = 10;
    public int input_size = 40;
    public int hidden_size = 100;
    public int output_size = 3;
    public float learning_rate_a = 1e-4f;
    public float learning_rate_v = 3e-4f;
    public float gamma = 0.99f;
    public float b = 2.0f;
    public float clip_param = 0.2f;
    public float max_grad_norm = 0.5f;
    public bool istraining = true;
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
    float losslabel = 0;
    // Update is called once per frame
    private void FixedUpdate()
    {
        int layerMask = 1 << 8;
        //layerMask = ~layerMask;
        float[,] s = new float[1, input_size];
        float r = 0;
        r = -0.6f;
        for (int i = -20; i < 20; i++)
        {
            Vector3 v = transform.TransformDirection(Vector3.forward);
            Vector3 rv = Quaternion.AngleAxis(i*4, Vector3.up)*v;
            
            RaycastHit hit;
            if (Physics.Raycast(transform.position, rv, out hit, Mathf.Infinity, layerMask))
            {
                s[0, (i + 20)*1] = hit.distance;
                if (hit.collider.name == "Banana" && hit.distance<3f)
                {
                    r += 0.5f;
                }
                if (hit.collider.name == "Banana")
                {
                    r += 0.4f;
                }

                // Debug.DrawRay(transform.position, rv * hit.distance, Color.yellow);
            }
            else
            {
               // Debug.DrawRay(transform.position, rv * 1000, Color.white);
            }
        }

        float[,] alp = new float[1, output_size];
        float[,] a = agentppo.Predict(s, out alp, b);

        if (Input.GetKey(KeyCode.LeftArrow))
        {
            a[0,0] = 15;
        }
        if (Input.GetKey(KeyCode.RightArrow))
        {
            a[0, 0] = 64;
        }
        r /= 40;
        transform.Rotate(new Vector3(0,a[0, 0], 0));
        transform.Translate(new Vector3(a[0, 1],0, a[0,2]));
        float x = Mathf.Clamp(transform.position.x, -5, 5);
        float y = Mathf.Clamp(transform.position.y, -5, 5);
        float z = Mathf.Clamp(transform.position.z, -5, 5);
        transform.position = new Vector3(x,y,z);
        GameObject.Find("Canvas").transform.Find("Text").GetComponent<Text>().text = r+"_"+losslabel;
        if (istraining == true)
        {
            float loss=agentppo.Train(s, a, r, alp);
            if (loss != 0)
                losslabel = loss * loss;
        }
    }
}

