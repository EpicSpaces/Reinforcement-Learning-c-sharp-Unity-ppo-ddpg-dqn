using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;
using System.Collections;

public class GridEnvironment_PPO_D : MonoBehaviour
{
    public float reward;
    public bool done;
    public int maxSteps;
    public int currentStep;
    public bool begun;
    public bool acceptingSteps;

    AgentPPO_D ppo_d;
    int batch_Capacity = 200;
    int batch_size = 32;
    int ppo_epoch = 10;
    int input_size = 100;
    int hidden_size = 20;
    int output_size = 4;
    float learning_rate_a = 1e-3f;
    float learning_rate_v = 3e-3f;
    float gamma = 0.99f;
    float clip_param = 0.2f;
    float max_grad_norm = 0.5f;
    
    public bool istraining = true;
    public float[] actions;
    public int episodeCount;
    public bool humanControl;

    public int bumper;

    public List<GameObject> actorObjs;
    public string[] players;
    public GameObject visualAgent;
    int numObstacles;
    int numGoals;
    int gridSize;
    int[] objectPositions;
    float episodeReward;

    void Start()
    {
        ppo_d = new AgentPPO_D(batch_Capacity,
    batch_size,
    ppo_epoch,
    input_size,
    hidden_size,
    output_size,
    learning_rate_a,
    learning_rate_v,
    gamma,
    clip_param,
    max_grad_norm);

        maxSteps = 100;
        int gridSizeSet = (GameObject.Find("Dropdown").GetComponent<Dropdown>().value + 1) * 5;
        numGoals = 1;
        numObstacles = Mathf.FloorToInt((gridSizeSet * gridSizeSet) / 10f);
        gridSize = gridSizeSet;

        foreach (GameObject actor in actorObjs)
        {
            DestroyImmediate(actor);
        }

        SetUp();
        Reset();

    }
    public void SetUp()
    {
        List<string> playersList = new List<string>();
        actorObjs = new List<GameObject>();
        for (int i = 0; i < numObstacles; i++)
        {
            playersList.Add("pit");
        }
        playersList.Add("agent");

        for (int i = 0; i < numGoals; i++)
        {
            playersList.Add("goal");
        }
        players = playersList.ToArray();
        Camera cam = GameObject.Find("Main Camera").GetComponent<Camera>();
        cam.transform.position = new Vector3((gridSize - 1), gridSize, -(gridSize - 1) / 2f);
        cam.orthographicSize = (gridSize + 5f) / 2f;
        SetEnvironment();
    }

    void Update()
    {
        if (acceptingSteps == true)
        {
            if (done == false)
            {
                Step();
            }
            else
            {
                Reset();
            }
        }
    }
    
    public List<float> collectState()
    {
        List<float> state = new List<float>();
        foreach (GameObject actor in actorObjs)
        {
            if (actor.tag == "agent")
            {
                float point = (gridSize * actor.transform.position.x) + actor.transform.position.z;
                state.Add(point);
            }
        }
        return state;
    }
    public void SetEnvironment()
    {
        GameObject.Find("Plane").transform.localScale = new Vector3(gridSize / 10.0f, 1f, gridSize / 10.0f);
        GameObject.Find("Plane").transform.position = new Vector3((gridSize - 1) / 2f, -0.5f, (gridSize - 1) / 2f);
        GameObject.Find("sN").transform.localScale = new Vector3(1, 1, gridSize + 2);
        GameObject.Find("sS").transform.localScale = new Vector3(1, 1, gridSize + 2);
        GameObject.Find("sN").transform.position = new Vector3((gridSize - 1) / 2f, 0.0f, gridSize);
        GameObject.Find("sS").transform.position = new Vector3((gridSize - 1) / 2f, 0.0f, -1);
        GameObject.Find("sE").transform.localScale = new Vector3(1, 1, gridSize + 2);
        GameObject.Find("sW").transform.localScale = new Vector3(1, 1, gridSize + 2);
        GameObject.Find("sE").transform.position = new Vector3(gridSize, 0.0f, (gridSize - 1) / 2f);
        GameObject.Find("sW").transform.position = new Vector3(-1, 0.0f, (gridSize - 1) / 2f);

        HashSet<int> numbers = new HashSet<int>();
        while (numbers.Count < players.Length)
        {
            numbers.Add(Random.Range(0, gridSize * gridSize));
        }
        objectPositions = numbers.ToArray();
    }
    public void LoadSpheres()
    {
        GameObject[] values = GameObject.FindGameObjectsWithTag("value");
        foreach (GameObject value in values)
        {
            Destroy(value);
        }

        /*float[,] value_estimates = adqn.net.Forward(s, a);
        for (int i = 0; i < gridSize * gridSize; i++)
        {
            GameObject value = (GameObject)GameObject.Instantiate(Resources.Load("value"));
            int x = i / gridSize;
            int y = i % gridSize;
            value.transform.position = new Vector3(x, 0.0f, y);
            value.transform.localScale = new Vector3(value_estimates[0,i] / 1.25f, value_estimates[0,i] / 1.25f, value_estimates[0,i] / 1.25f);
            if (value_estimates[0,i] < 0)
            {
                Material newMat = Resources.Load("negative_mat", typeof(Material)) as Material;
                value.GetComponent<Renderer>().material = newMat;
            }
        }*/
    }
    public virtual void Step()
    {
        acceptingSteps = false;
        currentStep += 1;
        if (currentStep >= maxSteps)
        {
            done = true;
        }
        float losslabel = 0;
        float[,] s = new float[1, input_size];
        reward = 0;
        GameObject actor =GameObject.FindGameObjectWithTag("agent");
        s[0, (int)(((10) * actor.transform.position.x) + actor.transform.position.z)] = (10 * actor.transform.position.x) + actor.transform.position.z;

        float ap;
        int act_idx = ppo_d.Predict(s, out ap);

        string actions = ""+act_idx;
        
        MiddleStep(actions);

        if (istraining == true)
        {
            float loss = ppo_d.Train(s, act_idx, reward,ap);
            if (loss != 0)
                losslabel = loss * loss;
        }
        acceptingSteps = true;
        GameObject.Find("RTxt").GetComponent<Text>().text = reward + "_" + losslabel;

    }
    public void Reset()
    {
        reward = 0;
        currentStep = 0;
        episodeCount++;
        done = false;
        acceptingSteps = false;

        foreach (GameObject actor in actorObjs)
        {
            DestroyImmediate(actor);
        }
        actorObjs = new List<GameObject>();

        for (int i = 0; i < players.Length; i++)
        {
            int x = (objectPositions[i]) / gridSize;
            int y = (objectPositions[i]) % gridSize;
            GameObject actorObj = (GameObject)GameObject.Instantiate(Resources.Load(players[i]));
            actorObj.transform.position = new Vector3(x, 0.0f, y);
            actorObj.name = players[i];
            actorObjs.Add(actorObj);
            if (players[i] == "agent")
            {
                visualAgent = actorObj;
            }
        }
        episodeReward = 0;
        if (istraining == true)
        {
//            float loss = addpg.Train(s, a, r);
  //          if (loss != 0)
    //            losslabel = loss * loss;
        }
        acceptingSteps = true;
        begun = true;

    }

    public void MiddleStep(string action)
    {
        reward = -0.05f;
        // 0 - Forward, 1 - Backward, 2 - Left, 3 - Right
		string s=action;
        //s =action+"";
        if (s.Contains("3"))
        {
            Collider[] blockTest = Physics.OverlapBox(new Vector3(visualAgent.transform.position.x + 1, 0, visualAgent.transform.position.z), new Vector3(0.3f, 0.3f, 0.3f));
            if (blockTest.Where(col => col.gameObject.tag == "wall").ToArray().Length == 0)
            {
                visualAgent.transform.position = new Vector3(visualAgent.transform.position.x + 1, 0, visualAgent.transform.position.z);
            }
        }

        if (s.Contains("2"))
        {
            Collider[] blockTest = Physics.OverlapBox(new Vector3(visualAgent.transform.position.x - 1, 0, visualAgent.transform.position.z), new Vector3(0.3f, 0.3f, 0.3f));
            if (blockTest.Where(col => col.gameObject.tag == "wall").ToArray().Length == 0)
            {
                visualAgent.transform.position = new Vector3(visualAgent.transform.position.x - 1, 0, visualAgent.transform.position.z);
            }
        }

        if (s.Contains("0"))
        {
            Collider[] blockTest = Physics.OverlapBox(new Vector3(visualAgent.transform.position.x, 0, visualAgent.transform.position.z + 1), new Vector3(0.3f, 0.3f, 0.3f));
            if (blockTest.Where(col => col.gameObject.tag == "wall").ToArray().Length == 0)
            {
                visualAgent.transform.position = new Vector3(visualAgent.transform.position.x, 0, visualAgent.transform.position.z + 1);
            }
        }

        if (s.Contains("1"))
        {
            Collider[] blockTest = Physics.OverlapBox(new Vector3(visualAgent.transform.position.x, 0, visualAgent.transform.position.z - 1), new Vector3(0.3f, 0.3f, 0.3f));
            if (blockTest.Where(col => col.gameObject.tag == "wall").ToArray().Length == 0)
            {
                visualAgent.transform.position = new Vector3(visualAgent.transform.position.x, 0, visualAgent.transform.position.z - 1);
            }
        }

        Collider[] hitObjects = Physics.OverlapBox(visualAgent.transform.position, new Vector3(0.3f, 0.3f, 0.3f));
        if (hitObjects.Where(col => col.gameObject.tag == "goal").ToArray().Length == 1)
        {
            reward = 1;
            done = true;
        }
        if (hitObjects.Where(col => col.gameObject.tag == "pit").ToArray().Length == 1)
        {
            reward = -1;
            done = true;
        }

        //LoadSpheres();
        episodeReward += reward;
        GameObject.Find("RTxt").GetComponent<Text>().text = "Episode Reward: " + episodeReward.ToString("F2");

    }
}
