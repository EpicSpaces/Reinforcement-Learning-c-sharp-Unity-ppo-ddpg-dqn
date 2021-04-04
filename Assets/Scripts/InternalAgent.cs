using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;

public class InternalAgent {
	public float[][] q_table;   // The matrix containing the values estimates.
	float learning_rate = 0.5f; // The rate at which to update the value estimates given a reward.
	int action = -1;
	float gamma = 0.99f; // Discount factor for calculating Q-target.
	float e = 1; // Initial epsilon value for random action selection.
	float eMin = 0.1f; // Lower bound of epsilon.
	int annealingSteps = 2000; // Number of steps to lower e to eMin.
	int lastState;

	string cases;
	public string[] cases_d;
	void recur(int[] arr, string outs, int i, int n, int k)
	{
		if (k > n)
			return;

		if (k == 0)
		{
			cases += outs + "\n";
			return;
		}

		for (int j = i; j < n; j++)
		{
			recur(arr, outs + " " + arr[j], j + 1, n, k - 1);
		}
	}
	public void SendParameters(EnvironmentParameters env)
	{
		q_table = new float[env.state_size][];
		action = 0;

		cases = "";
		int[] set = new int[env.action_size];
		for (int i = 0; i < set.Length; i++)
		{
			set[i] = i;
		}
		for (int i = 0; i < env.action_size; i++)
		{
			recur(set, "", 0, set.Length, i);
		}
		//Debug.Log(""+cases);
		int count = 1;
		int len = cases.Length - 1;
		for (int i = 0; i < len; ++i)
			switch (cases[i])
			{
				case '\r':
					if (cases[i + 1] == '\n')
					{
						if (++i >= len)
						{
							break;
						}
					}
					goto case '\n';
				case '\n':
					++count;
					break;
			}
		cases = cases.Remove(cases.Length - 1);
		cases = cases.Substring(1);
		cases = cases.Substring(1);
		cases_d = cases.Split('\n');
		for (int i = 0; i < env.state_size; i++) {
			q_table[i] = new float[/*env.action_size*/count - 1];
			for (int j = 0; j < env.action_size; j++) {
				q_table[i][j] = 0.0f;
			}
		}
	}

	/// <summary>
	/// Picks an action to take from its current state.
	/// </summary>
	/// <returns>The action choosen by the agent's policy</returns>
	public float[] GetAction() {
		int output_size=q_table[0].GetLength(0);
		int act_idx=0;
		float[] softmax = new float[output_size];

        float sumexp = 0;
        for (int i = 0; i < output_size; i++)
        {
            sumexp += Mathf.Exp(q_table[lastState][i]);
        }
        for (int i = 0; i < output_size; i++)
        {
            softmax[i] = Mathf.Exp(q_table[lastState][i]) / sumexp;
        }

        float[] weightsum = new float[output_size];
        
        weightsum[0] = softmax[0];

        for (int i = 1; i < output_size; i++)
        {
            weightsum[i] = weightsum[i - 1] + softmax[i];
        }
		for (int i = 0; i < output_size; i++)
        {
            weightsum[i] /= weightsum[output_size-1];
        }
        float r = Random.Range(0f, 1f);
        for (int i = 0; i < output_size; i++) 
		{
			if(r<weightsum[i])
			{
				act_idx=i;
				break;
			}
		}
        action=act_idx;
		//action = q_table[lastState].ToList().IndexOf(q_table[lastState].Max());
		//if (Random.Range(0f, 1f) < e) { action = Random.Range(0, 3); }
		//if (e > eMin) { e = e - ((1f - eMin) / (float)annealingSteps); }
		//GameObject.Find("ETxt").GetComponent<Text>().text = "Epsilon: " + e.ToString("F2");
		float currentQ = q_table[lastState][action];
		GameObject.Find("QTxt").GetComponent<Text>().text = "Current Q-value: " + currentQ.ToString("F2");
		return new float[1] { action };
	}

	/// <summary>
	/// Gets the values stored within the Q table.
	/// </summary>
	/// <returns>The average Q-values per state.</returns>
	public float[] GetValue() {
		float[] value_table = new float[q_table.Length];
		for (int i = 0; i < q_table.Length; i++)
		{
			value_table[i] = q_table[i].Average();
		}
		return value_table;
	}

	/// <summary>
	/// Updates the value estimate matrix given a new experience (state, action, reward).
	/// </summary>
	/// <param name="state">The environment state the experience happened in.</param>
	/// <param name="reward">The reward recieved by the agent from the environment for it's action.</param>
	/// <param name="done">Whether the episode has ended</param>
	public void SendState(List<float> state, float reward, bool done)
	{
		int nextState = Mathf.FloorToInt(state.First());
		if (action != -1) {
			if (done == true)
			{
				q_table[lastState][action] += learning_rate * (reward - q_table[lastState][action]);
			}
			else
			{
				q_table[lastState][action] += learning_rate * (reward + gamma * q_table[nextState].Max() - q_table[lastState][action]);
			}
		}
		lastState = nextState;
	}
}
