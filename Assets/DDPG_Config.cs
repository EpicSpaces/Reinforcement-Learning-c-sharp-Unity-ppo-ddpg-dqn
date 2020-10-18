using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DDPG_Config : MonoBehaviour
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
    public float b;
    public DDPG_Config(
    int batch_capacity = 2000,
    int batch_size = 32,
    int updater = 200,
    int input_size = 3,
    int hidden_size = 100,
    int output_size = 5,
    float learning_rate = 1e-3f,
    float max_grad_norm = 0.5f,
    float gamma = 0.99f,
    float epsilon = 1f,
    float decay = 0.999f,
    float min_decay = 0.01f,
    float b =2.0f)
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
        this.b = b;
    }
    public DDPG_Config(DDPG_Config ddpg_config)
    {
        this.batch_capacity = ddpg_config.batch_capacity;
        this.batch_size = ddpg_config.batch_size;
        this.updater = ddpg_config.updater;
        this.input_size = ddpg_config.input_size;
        this.hidden_size = ddpg_config.hidden_size;
        this.output_size =ddpg_config.output_size;
        this.learning_rate = ddpg_config.learning_rate;
        this.max_grad_norm = ddpg_config.max_grad_norm;
        this.gamma = ddpg_config.gamma;
        this.epsilon = ddpg_config.epsilon;
        this.decay = ddpg_config.decay;
        this.min_decay = ddpg_config.min_decay;
        this.b = ddpg_config.b;
    }
}
