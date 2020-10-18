using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DQN_Config : MonoBehaviour
{
    public int batch_capacity;
    public int batch_size;
    public int updater;
    public int input_size;
    public int hidden_size;
    public int output_size;
    public float learning_rate;
    public float max_grad_norm;
    public float gamma;
    public float epsilon;
    public float decay;
    public float min_decay;
    public DQN_Config(
    int batch_capacity = 2000,
    int batch_size = 32,
    int updater = 30,
    int input_size = 3,
    int hidden_size = 100,
    int output_size = 5,
    float learning_rate = 1e-3f,
    float max_grad_norm = 0.5f,
    float gamma = 0.99f,
    float epsilon = 1f,
    float decay = 0.999f,
    float min_decay = 0.01f)
    {
        this.batch_capacity = batch_capacity;
        this.batch_size = batch_size;
        this.updater = updater;
        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.output_size = output_size;
        this.learning_rate = learning_rate;
        this.max_grad_norm = max_grad_norm;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.decay = decay;
        this.min_decay = min_decay;
    }
    public DQN_Config(DQN_Config dqn_config)
    {
        this.batch_capacity = dqn_config.batch_capacity;
        this.batch_size = dqn_config.batch_size;
        this.updater = dqn_config.updater;
        this.input_size = dqn_config.input_size;
        this.hidden_size = dqn_config.hidden_size;
        this.output_size =dqn_config.output_size;
        this.learning_rate = dqn_config.learning_rate;
        this.max_grad_norm = dqn_config.max_grad_norm;
        this.gamma = dqn_config.gamma;
        this.epsilon = dqn_config.epsilon;
        this.decay = dqn_config.decay;
        this.min_decay = dqn_config.min_decay;
    }
}
