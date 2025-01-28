#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// 定义激活函数类型
typedef enum {
    RELU,
    SIGMOID,
    TANH,
    SOFTMAX
} Activation;

// 定义神经网络层
typedef struct {
    int input_size;
    int output_size;
    double **weights;
    double *biases;
    double *input;
    double *output;
    double *deltas;
    Activation activation;

    // Adam优化器变量
    double **weight_m; // 一阶矩估计
    double **weight_v; // 二阶矩估计
    double *bias_m;
    double *bias_v;

    // Dropout相关变量
    double dropout_probability; // Dropout概率
    int *dropout_mask;          // Dropout掩码（1表示保留，0表示丢弃）
} Layer;

// 定义神经网络
typedef struct {
    int num_layers;
    Layer *layers;
    double learning_rate;

    // Adam优化器超参数
    double beta1;
    double beta2;
    double epsilon;
    int t; // 时间步

    // L2 正则化参数
    double lambda; // 正则化强度

    // 训练模式标志
    int is_training; // 1表示训练，0表示推理
} NeuralNetwork;

// 定义训练样本结构体
typedef struct {
    double *input;
    double *label;
} Sample;

// 安全内存分配函数
void* safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "内存分配失败。\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// 分配二维数组
double** allocate_2d_array(int rows, int cols) {
    double **array = (double**)safe_malloc(rows * sizeof(double*));
    for(int i = 0; i < rows; i++) {
        array[i] = (double*)safe_malloc(cols * sizeof(double));
    }
    return array;
}

// 将激活函数枚举转换为字符串
const char* activation_to_string(Activation a) {
    switch(a) {
        case RELU:
            return "RELU";
        case SIGMOID:
            return "SIGMOID";
        case TANH:
            return "TANH";
        case SOFTMAX:
            return "SOFTMAX";
        default:
            return "UNKNOWN";
    }
}

// 将字符串转换为激活函数枚举
Activation string_to_activation(const char *s) {
    if(strcmp(s, "RELU") == 0)
        return RELU;
    if(strcmp(s, "SIGMOID") == 0)
        return SIGMOID;
    if(strcmp(s, "TANH") == 0)
        return TANH;
    if(strcmp(s, "SOFTMAX") == 0)
        return SOFTMAX;
    // 默认返回 RELU
    return RELU;
}

// 初始化Layer
Layer initialize_layer(int input_size, int output_size, Activation activation, double dropout_prob) {
    Layer layer;
    layer.input_size = input_size;
    layer.output_size = output_size;
    layer.activation = activation;
    layer.dropout_probability = dropout_prob;

    // 初始化权重（He 初始化）
    layer.weights = allocate_2d_array(output_size, input_size);
    for(int i = 0; i < output_size; i++) {
        for(int j = 0; j < input_size; j++) {
            double stddev = sqrt(2.0 / input_size);
            layer.weights[i][j] = ((double)rand() / RAND_MAX) * 2 * stddev - stddev; // 初始化为 [-stddev, stddev)
        }
    }

    // 初始化偏置
    layer.biases = (double*)safe_malloc(output_size * sizeof(double));
    for(int i = 0; i < output_size; i++) {
        layer.biases[i] = 0.0;
    }

    // 初始化激活相关变量
    layer.input = (double*)safe_malloc(input_size * sizeof(double));
    layer.output = (double*)safe_malloc(output_size * sizeof(double));
    layer.deltas = (double*)calloc(output_size, sizeof(double));
    if (!layer.input || !layer.output || !layer.deltas) {
        fprintf(stderr, "内存分配失败：Layer激活。\n");
        exit(EXIT_FAILURE);
    }

    // 初始化Adam优化器变量
    layer.weight_m = allocate_2d_array(output_size, input_size);
    layer.weight_v = allocate_2d_array(output_size, input_size);
    layer.bias_m = (double*)safe_malloc(output_size * sizeof(double));
    layer.bias_v = (double*)safe_malloc(output_size * sizeof(double));
    for(int i = 0; i < output_size; i++) {
        layer.bias_m[i] = 0.0;
        layer.bias_v[i] = 0.0;
        for(int j = 0; j < input_size; j++) {
            layer.weight_m[i][j] = 0.0;
            layer.weight_v[i][j] = 0.0;
        }
    }

    // 初始化Dropout掩码
    layer.dropout_mask = (int*)safe_malloc(output_size * sizeof(int));
    for(int i = 0; i < output_size; i++) {
        layer.dropout_mask[i] = 1; // 初始时全部保留
    }

    return layer;
}

// 创建神经网络
NeuralNetwork* create_network(int num_layers, int *layers_sizes, Activation *activations, double lambda, double dropout_prob, double learning_rate) {
    NeuralNetwork *nn = (NeuralNetwork*)safe_malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers - 1; // 不包括输入层
    nn->layers = (Layer*)safe_malloc(nn->num_layers * sizeof(Layer));

    for(int i = 0; i < nn->num_layers; i++) {
        double current_dropout = (activations[i] != SOFTMAX) ? dropout_prob : 0.0; // 输出层不应用Dropout
        nn->layers[i] = initialize_layer(layers_sizes[i], layers_sizes[i+1], activations[i], current_dropout);
    }

    // 初始化Adam优化器超参数
    nn->beta1 = 0.9;
    nn->beta2 = 0.999;
    nn->epsilon = 1e-8;
    nn->t = 0;

    // 设置L2正则化参数
    nn->lambda = lambda;

    // 设置训练模式
    nn->is_training = 1;

    // 设置学习率
    nn->learning_rate = learning_rate;

    return nn;
}

// 保存神经网络到二进制文件
int save_network(NeuralNetwork *nn, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if(!fp){
        fprintf(stderr, "无法打开文件进行保存：%s\n", filename);
        return -1;
    }

    // 保存 NeuralNetwork 结构
    fwrite(&nn->num_layers, sizeof(int), 1, fp);
    fwrite(&nn->learning_rate, sizeof(double), 1, fp);
    fwrite(&nn->beta1, sizeof(double), 1, fp);
    fwrite(&nn->beta2, sizeof(double), 1, fp);
    fwrite(&nn->epsilon, sizeof(double), 1, fp);
    fwrite(&nn->t, sizeof(int), 1, fp);
    fwrite(&nn->lambda, sizeof(double), 1, fp);

    // 保存每一层的参数
    for(int l=0; l < nn->num_layers; l++){
        Layer *layer = &nn->layers[l];
        fwrite(&layer->input_size, sizeof(int), 1, fp);
        fwrite(&layer->output_size, sizeof(int), 1, fp);
        fwrite(&layer->activation, sizeof(Activation), 1, fp);
        fwrite(&layer->dropout_probability, sizeof(double), 1, fp);

        // 保存权重
        for(int i=0; i < layer->output_size; i++){
            fwrite(layer->weights[i], sizeof(double), layer->input_size, fp);
        }

        // 保存偏置
        fwrite(layer->biases, sizeof(double), layer->output_size, fp);

        // 保存 Adam 优化器变量
        for(int i=0; i < layer->output_size; i++){
            fwrite(layer->weight_m[i], sizeof(double), layer->input_size, fp);
            fwrite(layer->weight_v[i], sizeof(double), layer->input_size, fp);
        }
        fwrite(layer->bias_m, sizeof(double), layer->output_size, fp);
        fwrite(layer->bias_v, sizeof(double), layer->output_size, fp);
    }

    fclose(fp);
    printf("模型已保存到二进制文件：%s\n", filename);
    return 0;
}

// 加载神经网络从二进制文件
NeuralNetwork* load_network(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if(!fp){
        fprintf(stderr, "无法打开文件进行加载：%s\n", filename);
        return NULL;
    }

    // 读取 NeuralNetwork 结构
    NeuralNetwork *nn = (NeuralNetwork*)safe_malloc(sizeof(NeuralNetwork));
    fread(&nn->num_layers, sizeof(int), 1, fp);
    fread(&nn->learning_rate, sizeof(double), 1, fp);
    fread(&nn->beta1, sizeof(double), 1, fp);
    fread(&nn->beta2, sizeof(double), 1, fp);
    fread(&nn->epsilon, sizeof(double), 1, fp);
    fread(&nn->t, sizeof(int), 1, fp);
    fread(&nn->lambda, sizeof(double), 1, fp);
    nn->is_training = 1; // 默认设置为训练模式

    // 分配内存给层
    nn->layers = (Layer*)safe_malloc(nn->num_layers * sizeof(Layer));

    // 读取每一层的参数
    for(int l=0; l < nn->num_layers; l++){
        Layer *layer = &nn->layers[l];
        fread(&layer->input_size, sizeof(int), 1, fp);
        fread(&layer->output_size, sizeof(int), 1, fp);
        fread(&layer->activation, sizeof(Activation), 1, fp);
        fread(&layer->dropout_probability, sizeof(double), 1, fp);

        // 分配并读取权重
        layer->weights = allocate_2d_array(layer->output_size, layer->input_size);
        for(int i=0; i < layer->output_size; i++){
            fread(layer->weights[i], sizeof(double), layer->input_size, fp);
        }

        // 分配并读取偏置
        layer->biases = (double*)safe_malloc(layer->output_size * sizeof(double));
        fread(layer->biases, sizeof(double), layer->output_size, fp);

        // 初始化激活相关变量
        layer->input = (double*)safe_malloc(layer->input_size * sizeof(double));
        layer->output = (double*)safe_malloc(layer->output_size * sizeof(double));
        layer->deltas = (double*)calloc(layer->output_size, sizeof(double));
        if (!layer->input || !layer->output || !layer->deltas) {
            fprintf(stderr, "内存分配失败：Layer激活。\n");
            exit(EXIT_FAILURE);
        }

        // 分配并读取 Adam 优化器变量
        layer->weight_m = allocate_2d_array(layer->output_size, layer->input_size);
        layer->weight_v = allocate_2d_array(layer->output_size, layer->input_size);
        layer->bias_m = (double*)safe_malloc(layer->output_size * sizeof(double));
        layer->bias_v = (double*)safe_malloc(layer->output_size * sizeof(double));
        for(int i=0; i < layer->output_size; i++) {
            fread(layer->weight_m[i], sizeof(double), layer->input_size, fp);
            fread(layer->weight_v[i], sizeof(double), layer->input_size, fp);
        }
        fread(layer->bias_m, sizeof(double), layer->output_size, fp);
        fread(layer->bias_v, sizeof(double), layer->output_size, fp);

        // 初始化Dropout掩码
        layer->dropout_mask = (int*)safe_malloc(layer->output_size * sizeof(int));
        for(int i = 0; i < layer->output_size; i++) {
            layer->dropout_mask[i] = 1; // 初始时全部保留
        }
    }

    fclose(fp);
    printf("模型已从二进制文件 %s 加载成功。\n", filename);
    return nn;
}

// 释放神经网络内存
void free_network_mem(NeuralNetwork *nn) {
    for(int l = 0; l < nn->num_layers; l++) {
        for(int i = 0; i < nn->layers[l].output_size; i++) {
            free(nn->layers[l].weights[i]);
            free(nn->layers[l].weight_m[i]);
            free(nn->layers[l].weight_v[i]);
        }
        free(nn->layers[l].weights);
        free(nn->layers[l].weight_m);
        free(nn->layers[l].weight_v);
        free(nn->layers[l].biases);
        free(nn->layers[l].bias_m);
        free(nn->layers[l].bias_v);
        free(nn->layers[l].input);
        free(nn->layers[l].output);
        free(nn->layers[l].deltas);
        free(nn->layers[l].dropout_mask); // 释放Dropout掩码
    }
    free(nn->layers);
    free(nn);
}

// 前向传播
void forward_propagation(NeuralNetwork *nn, double *input) {
    // 设置输入层
    for(int i = 0; i < nn->layers[0].input_size; i++) {
        nn->layers[0].input[i] = input[i];
    }

    // 遍历每一层
    for(int l = 0; l < nn->num_layers; l++) {
        Layer *current = &nn->layers[l];
        double *prev_output;
        if(l == 0) {
            prev_output = current->input;
        } else {
            prev_output = nn->layers[l-1].output;
        }

        // 计算加权和 + 偏置
        for(int i = 0; i < current->output_size; i++) {
            double sum = 0.0;
            for(int j = 0; j < current->input_size; j++) {
                sum += current->weights[i][j] * prev_output[j];
            }
            sum += current->biases[i];
            current->output[i] = sum;
        }

        // 应用激活函数
        switch(current->activation) {
            case RELU: {
                for(int i = 0; i < current->output_size; i++) {
                    current->output[i] = (current->output[i] > 0) ? current->output[i] : 0.0;
                }
                break;
            }
            case SIGMOID: {
                for(int i = 0; i < current->output_size; i++) {
                    current->output[i] = 1.0 / (1.0 + exp(-current->output[i]));
                }
                break;
            }
            case TANH: {
                for(int i = 0; i < current->output_size; i++) {
                    current->output[i] = tanh(current->output[i]);
                }
                break;
            }
            case SOFTMAX: {
                double max = current->output[0];
                for(int i = 1; i < current->output_size; i++) {
                    if(current->output[i] > max) max = current->output[i];
                }

                double sum = 0.0;
                for(int i = 0; i < current->output_size; i++) {
                    current->output[i] = exp(current->output[i] - max); // 防止溢出
                    sum += current->output[i];
                }

                for(int i = 0; i < current->output_size; i++) {
                    current->output[i] /= sum;
                }
                break;
            }
            default:
                break;
        }

        // 应用 Dropout（仅在训练模式下）
        if(current->dropout_probability > 0.0 && nn->is_training) {
            for(int i = 0; i < current->output_size; i++) {
                double rand_val = ((double)rand() / RAND_MAX);
                if(rand_val < current->dropout_probability) {
                    current->dropout_mask[i] = 0;          // 丢弃神经元
                    current->output[i] = 0.0;              // 设置输出为0
                } else {
                    current->dropout_mask[i] = 1;          // 保留神经元
                    current->output[i] /= (1.0 - current->dropout_probability); // 缩放输出
                }
            }
        }
    }
}

// 计算损失（交叉熵 + L2 正则化）
double compute_loss(NeuralNetwork *nn, double *expected_output) {
    Layer *output_layer = &nn->layers[nn->num_layers -1];
    double loss = 0.0;
    for(int i = 0; i < output_layer->output_size; i++) {
        loss -= expected_output[i] * log(output_layer->output[i] + 1e-15); // 防止log(0)
    }

    // 添加L2正则化项
    double l2_sum = 0.0;
    for(int l = 0; l < nn->num_layers; l++) {
        Layer *layer = &nn->layers[l];
        for(int i = 0; i < layer->output_size; i++) {
            for(int j = 0; j < layer->input_size; j++) {
                l2_sum += layer->weights[i][j] * layer->weights[i][j];
            }
        }
    }
    loss += (nn->lambda / 2.0) * l2_sum;

    return loss;
}

// 反向传播
void backward_propagation(NeuralNetwork *nn, double *expected_output) {
    // 从输出层开始
    for(int l = nn->num_layers -1; l >=0; l--) {
        Layer *current = &nn->layers[l];
        Layer *prev_layer = (l == 0) ? NULL : &nn->layers[l-1];

        if(current->activation == SOFTMAX) {
            // 对于Softmax和交叉熵损失，梯度为预测值 - 真实值
            for(int i = 0; i < current->output_size; i++) {
                current->deltas[i] = current->output[i] - expected_output[i];
            }
        }
        else {
            for(int i = 0; i < current->output_size; i++) {
                double derivative = 1.0;
                switch(current->activation) {
                    case RELU:
                        derivative = (current->output[i] > 0) ? 1.0 : 0.0;
                        break;
                    case SIGMOID:
                        derivative = current->output[i] * (1.0 - current->output[i]);
                        break;
                    case TANH:
                        derivative = 1.0 - current->output[i] * current->output[i];
                        break;
                    default:
                        break;
                }

                double delta_sum = 0.0;
                if(l < nn->num_layers -1) { // 如果不是最后一层
                    Layer *next_layer = &nn->layers[l+1];
                    for(int j = 0; j < next_layer->output_size; j++) {
                        delta_sum += next_layer->weights[j][i] * next_layer->deltas[j];
                    }
                }

                // 应用 Dropout 掩码（仅在训练模式下）
                if(current->dropout_probability > 0.0 && nn->is_training) {
                    delta_sum *= current->dropout_mask[i];
                }

                current->deltas[i] = delta_sum * derivative;
            }
        }
    }
}

// 更新参数使用Adam优化器，并添加自动学习率衰减
void update_parameters(NeuralNetwork *nn) {
    nn->t += 1; // 增加时间步

    // Adam的学习率校正
    double lr_t = nn->learning_rate * sqrt(1.0 - pow(nn->beta2, nn->t)) / (1.0 - pow(nn->beta1, nn->t));

    for(int l = 0; l < nn->num_layers; l++) {
        Layer *current = &nn->layers[l];
        double *prev_output = (l == 0) ? current->input : nn->layers[l-1].output;
        for(int i = 0; i < current->output_size; i++) {
            for(int j = 0; j < current->input_size; j++) {
                // 计算权重梯度，加上L2正则化
                double grad = current->deltas[i] * prev_output[j] + nn->lambda * current->weights[i][j];

                // 更新一阶矩估计
                current->weight_m[i][j] = nn->beta1 * current->weight_m[i][j] + (1.0 - nn->beta1) * grad;
                // 更新二阶矩估计
                current->weight_v[i][j] = nn->beta2 * current->weight_v[i][j] + (1.0 - nn->beta2) * (grad * grad);
                // 更新权重
                current->weights[i][j] -= lr_t * current->weight_m[i][j] / (sqrt(current->weight_v[i][j]) + nn->epsilon);
            }
            // 计算偏置梯度（不包括L2正则化）
            double grad_bias = current->deltas[i];
            // 更新一阶矩估计
            current->bias_m[i] = nn->beta1 * current->bias_m[i] + (1.0 - nn->beta1) * grad_bias;
            // 更新二阶矩估计
            current->bias_v[i] = nn->beta2 * current->bias_v[i] + (1.0 - nn->beta2) * (grad_bias * grad_bias);
            // 更新偏置
            current->biases[i] -= lr_t * current->bias_m[i] / (sqrt(current->bias_v[i]) + nn->epsilon);
        }
    }
}

// 创建训练数据
Sample* prepare_training_data(int training_set_size, int input_size, int output_size) {
    Sample *training_data = (Sample*)safe_malloc(training_set_size * sizeof(Sample));
    for(int num = 0; num < training_set_size; num++) {
        // 将整数转换为二进制表示
        training_data[num].input = (double*)safe_malloc(input_size * sizeof(double));
        for(int i = 0; i < input_size; i++) {
            training_data[num].input[input_size - 1 - i] = ((num >> i) & 1) ? 1.0 : 0.0;
        }

        // 生成标签：偶数 -> [1.0, 0.0], 奇数 -> [0.0, 1.0]
        training_data[num].label = (double*)safe_malloc(output_size * sizeof(double));
        if(num % 2 == 0) {
            training_data[num].label[0] = 1.0;
            training_data[num].label[1] = 0.0;
        }
        else {
            training_data[num].label[0] = 0.0;
            training_data[num].label[1] = 1.0;
        }
    }
    return training_data;
}

// 评估准确率
double evaluate_accuracy(NeuralNetwork *nn, Sample *training_data, int size) {
    int correct = 0;
    nn->is_training = 0; // 设置为推理模式
    for(int i = 0; i < size; i++) {
        forward_propagation(nn, training_data[i].input);

        Layer *output_layer = &nn->layers[nn->num_layers -1];
        int predicted_class = 0;
        double max_prob = output_layer->output[0];
        for(int j = 1; j < output_layer->output_size; j++) {
            if(output_layer->output[j] > max_prob) {
                max_prob = output_layer->output[j];
                predicted_class = j;
            }
        }

        // 真实类别
        int true_class = (training_data[i].label[0] == 1.0) ? 0 : 1;

        if(predicted_class == true_class) {
            correct++;
        }
    }
    nn->is_training = 1; // 恢复训练模式
    return ((double)correct) / size;
}

// 创建二进制输入
double* create_binary_input(int num, int input_size) {
    double *binary_input = (double*)safe_malloc(input_size * sizeof(double));
    for(int i = 0; i < input_size; i++) {
        binary_input[input_size - 1 - i] = ((num >> i) & 1) ? 1.0 : 0.0;
    }
    return binary_input;
}

// 预测类别
int predict(NeuralNetwork *nn, double *binary_input) {
    forward_propagation(nn, binary_input);

    Layer *output_layer = &nn->layers[nn->num_layers -1];
    int predicted_class = 0;
    double max_prob = output_layer->output[0];
    for(int i = 1; i < output_layer->output_size; i++) {
        if(output_layer->output[i] > max_prob) {
            max_prob = output_layer->output[i];
            predicted_class = i;
        }
    }
    return predicted_class;
}

// 更新模型（一次训练步）
void update_model(NeuralNetwork *nn, double *binary_input, double *expected_output) {
    forward_propagation(nn, binary_input);
    backward_propagation(nn, expected_output);
    update_parameters(nn);
}

// 简化的训练函数
void train_network(NeuralNetwork *nn, Sample *training_data, int training_set_size, int training_epochs, int decay_step, double decay_rate) {
    printf("开始训练BP神经网络用于奇数/偶数分类...\n");
    for(int epoch = 1; epoch <= training_epochs; epoch++) {
        double total_loss = 0.0;

        // 遍历所有训练样本
        for(int i = 0; i < training_set_size; i++) {
            // 前向传播
            forward_propagation(nn, training_data[i].input);

            // 计算损失
            double loss = compute_loss(nn, training_data[i].label);
            total_loss += loss;

            // 反向传播
            backward_propagation(nn, training_data[i].label);

            // 更新参数（使用Adam优化器）
            update_parameters(nn);
        }

        // 计算平均损失和准确率
        double average_loss = total_loss / training_set_size;
        double accuracy = evaluate_accuracy(nn, training_data, training_set_size);

        // 每100个epoch输出一次训练状态
        if(epoch % 100 == 0 || epoch == 1) {
            printf("Epoch %d/%d - 平均损失: %.6f - 准确率: %.2f%% - 学习率: %.5f\n", 
                   epoch, training_epochs, average_loss, accuracy * 100, nn->learning_rate);
        }

        // 如果达到衰减步长，自动衰减学习率
        if(epoch % decay_step == 0) {
            nn->learning_rate *= decay_rate;
            printf("学习率已衰减至: %.5f\n", nn->learning_rate);
        }

        // 如果准确率达到100%，提前停止训练
        if(accuracy == 1.0) {
            printf("训练在第 %d 个周期时达到100%%准确率，提前停止。\n", epoch);
            break;
        }
    }

    printf("训练完成！\n\n");
}

// 用户交互模式
void user_interaction(NeuralNetwork *nn) {
    printf("\n奇数/偶数分类神经网络已准备完成。\n");
    printf("您可以输入一个整数进行分类，输入'exit'退出程序。\n");

    char input_str[100];
    while(1) {
        printf("请输入一个整数（或输入'exit'退出）：");
        if (scanf("%s", input_str) != 1) {
            printf("无效的输入，请重新输入。\n");
            // 清除输入缓冲区
            int c;
            while ((c = getchar()) != '\n' && c != EOF);
            continue;
        }

        // 检查用户是否想退出
        if (strcmp(input_str, "exit") == 0) {
            break;
        }

        // 将输入转换为整数
        int num = atoi(input_str);

        // 创建二进制输入
        int input_size = nn->layers[0].input_size;
        double *binary_input = create_binary_input(num, input_size);

        // 预测类别
        int predicted_class = predict(nn, binary_input);
        double predicted_prob = (predicted_class == 0) ? nn->layers[nn->num_layers -1].output[0] : nn->layers[nn->num_layers -1].output[1];

        if(predicted_class == 0) {
            printf("预测结果: 偶数 (概率: %.2f%%)\n", predicted_prob * 100);
        } else {
            printf("预测结果: 奇数 (概率: %.2f%%)\n", predicted_prob * 100);
        }

        // 提供反馈，是否进一步训练
        printf("是否要用这个样本继续训练？(y/n): ");
        char train_choice;
        scanf(" %c", &train_choice);
        if(train_choice == 'y' || train_choice == 'Y') {
            // 生成期望输出
            double *expected_output = (double*)safe_malloc(nn->layers[nn->num_layers -1].output_size * sizeof(double));
            if(num % 2 == 0) {
                expected_output[0] = 1.0;
                expected_output[1] = 0.0;
            }
            else {
                expected_output[0] = 0.0;
                expected_output[1] = 1.0;
            }

            // 更新模型
            update_model(nn, binary_input, expected_output);
            printf("已使用该样本进行一次训练更新。\n");

            free(expected_output);
        }

        free(binary_input);
        printf("\n");
    }
}

// 主函数
int main() {
    srand(time(NULL));

    // 定义网络结构
    // 示例：输入层(8) -> 隐藏层1(16, RELU) -> 隐藏层2(32, RELU) -> 隐藏层3(16, RELU) -> 输出层(2, SOFTMAX)
    int num_layers = 5;
    int layers_sizes[] = {8, 16, 32, 16, 2};
    Activation activations[] = {RELU, RELU, RELU, SOFTMAX};
    double initial_learning_rate = 0.01;
    double lambda = 0.001; // L2 正则化强度
    double dropout_probability = 0.3; // Dropout概率（降低至0.3）

    // 定义学习率衰减参数
    double learning_rate_decay = 0.1; // 每次衰减为原来的10%
    int learning_rate_decay_step = 100; // 每100个周期衰减一次

    // 提供加载模型的选项
    printf("是否要加载一个已有的模型？(y/n): ");
    char load_choice;
    scanf(" %c", &load_choice);
    NeuralNetwork *nn = NULL;
    Sample *training_data = NULL;

    if(load_choice == 'y' || load_choice == 'Y') {
        char load_filename[256];
        printf("请输入模型文件名：");
        scanf("%s", load_filename);
        nn = load_network(load_filename);
        if(nn == NULL){
            printf("加载模型失败，程序将退出。\n");
            return EXIT_FAILURE;
        }
    }
    else {
        // 创建网络
        nn = create_network(num_layers, layers_sizes, activations, lambda, dropout_probability, initial_learning_rate);

        // 准备训练数据
        int training_set_size = 256;
        int input_size = layers_sizes[0];
        int output_size = layers_sizes[num_layers -1];
        training_data = prepare_training_data(training_set_size, input_size, output_size);
    }

    // 训练参数
    int training_epochs = 1000;

    // 训练过程
    if(!(load_choice == 'y' || load_choice == 'Y')){
        train_network(nn, training_data, 256, training_epochs, learning_rate_decay_step, learning_rate_decay);
    }

    // 提供保存模型的选项
    printf("是否要保存当前模型？(y/n): ");
    char save_choice;
    scanf(" %c", &save_choice);
    if(save_choice == 'y' || save_choice == 'Y') {
        char save_filename[256];
        printf("请输入要保存的模型文件名：");
        scanf("%s", save_filename);
        save_network(nn, save_filename);
    }

    // 进入用户交互模式
    user_interaction(nn);

    // 如果未加载模型，释放训练数据内存
    if(!(load_choice == 'y' || load_choice == 'Y')){
        for(int i = 0; i < 256; i++) {
            free(training_data[i].input);
            free(training_data[i].label);
        }
        free(training_data);
    }

    // 释放网络内存
    free_network_mem(nn);

    return 0;
}
