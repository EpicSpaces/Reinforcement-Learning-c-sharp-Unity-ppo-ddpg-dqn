using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

public class Agent4444 : MonoBehaviour
{
    AgentDDPG addpg;
    DDPG_Config ddpg_config;
    public int batch_capacity = 2000;
    public int batch_size = 32;
    public int updater = 200;
    public int input_size = 20;
    public int hidden_size = 100;
    public int output_size = 4;
    public float learning_rate = 1e-3f;
    public float max_grad_norm = 0.5f;
    public float gamma = 0.99f;
    public float epsilon = 1f;
    public float decay = 0.999f;
    public float min_decay = 0.01f;
    public float b = 2.0f;
    public bool istraining = true;
    void Start()
    {
        ddpg_config = new DDPG_Config(
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
        min_decay,
        b);
        addpg = new AgentDDPG(ddpg_config);
    }
    float losslabel=0;
    // Update is called once per frame
    private void FixedUpdate()
    {
        int layerMask = 1 << 8;
        //layerMask = ~layerMask;
        float[,] s = new float[1, 2 * 20];
        float r = 0;
        bool shoot = false;
        r = -0.6f;
        for (int i = -20; i < 20; i++)
        {
            Vector3 v = transform.TransformDirection(Vector3.forward);
            Vector3 rv = Quaternion.AngleAxis(i * 4, Vector3.up) * v;
            Debug.DrawRay(transform.position, 10 * rv, Color.green);
        
            RaycastHit hit;
            if (Physics.Raycast(transform.position, rv, out hit, Mathf.Infinity, layerMask))
            {
                s[0, (i + 20) * 1] = hit.distance;
                if (hit.collider.name == "Banana" && hit.distance < 3f)
                {
                    r += 0.5f;
                }
                if (hit.collider.name == "Banana" && i == 0)
                {
                    r += 0.4f;
                    shoot = true;
                }

                Debug.DrawRay(transform.position, rv * hit.distance, Color.yellow);
            }
            else
            {
                Debug.DrawRay(transform.position, rv * 1000, Color.white);
            }
        }
        float[,] a = addpg.Predict(s);
        if (Input.GetKey(KeyCode.LeftArrow))
        {
            a[0, 0] = 15;
        }
        if (Input.GetKey(KeyCode.RightArrow))
        {
            a[0, 1] = 64;
        }

        if (shoot == true && a[0, 3]<2&& a[0, 3] >1.9f)
            r += 0.3f;
        r /= 40;

        transform.Translate(new Vector3(a[0, 0], 0, a[0, 1]));
        transform.Rotate(new Vector3(0, a[0, 2], 0));
        if(a[0, 3] < 2 && a[0, 3] > 1.9f)
        Debug.DrawRay(transform.position, 100 * transform.TransformDirection(Vector3.up), Color.red);
        
        float x = Mathf.Clamp(transform.position.x, -5, 5);
        float y = Mathf.Clamp(transform.position.y, -5, 5);
        float z = Mathf.Clamp(transform.position.z, -5, 5);
        transform.position = new Vector3(x, y, z);

        GameObject.Find("Canvas").transform.Find("Text").GetComponent<Text>().text = r + "_" + losslabel;
        if (istraining == true)
        {
            float loss = addpg.Train(s, a, r);
            if (loss != 0)
                losslabel = loss * loss;
        }
    }
}
