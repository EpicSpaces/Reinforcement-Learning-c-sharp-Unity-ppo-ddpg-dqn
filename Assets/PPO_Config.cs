using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PPO_Config : MonoBehaviour
{
    public int batch_Capacity;
    public int batch_size;
    public int ppo_epoch;
    public int input_size;
    public int hidden_size;
    public int output_size;
    public float learning_rate_a;
    public float learning_rate_v;
    public float gamma;
    public float b;
    public float clip_param;
    public float max_grad_norm;
    public PPO_Config(
        int batch_Capacity = 1000,
        int batch_size = 32,
        int ppo_epoch = 10,
        int input_size = 3,
        int hidden_size = 100,
        int output_size = 1,
        float learning_rate_a = 1e-4f,
        float learning_rate_v = 3e-4f,
        float gamma = 0.99f,
        float b = 2.0f,
        float clip_param = 0.2f,
        float max_grad_norm = 0.5f)
    {
        this.batch_Capacity = batch_Capacity;
        this.batch_size = batch_size;
        this.ppo_epoch = ppo_epoch;
        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.output_size = output_size;
        this.learning_rate_a = learning_rate_a;
        this.learning_rate_v = learning_rate_v;
        this.gamma = gamma;
        this.b = b;
        this.clip_param =clip_param;
        this.max_grad_norm = max_grad_norm;
        
    }
    public PPO_Config(PPO_Config dqn_config)
    {
        this.batch_Capacity = dqn_config.batch_Capacity;
        this.batch_size = dqn_config.batch_size;
        this.ppo_epoch = dqn_config.ppo_epoch;
        this.input_size = dqn_config.input_size;
        this.hidden_size = dqn_config.hidden_size;
        this.output_size = dqn_config.output_size;
        this.learning_rate_a = dqn_config.learning_rate_a;
        this.learning_rate_v = dqn_config.learning_rate_v;
        this.gamma = dqn_config.gamma;
        this.b = dqn_config.b;
        this.clip_param = dqn_config.clip_param;
        this.max_grad_norm = dqn_config.max_grad_norm;

    }

}
