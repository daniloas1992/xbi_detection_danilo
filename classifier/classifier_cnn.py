from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AvgPool2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import SGD


def get_params_grid_cnn():
    # Batch Size: Número de padrões mostrados à rede antes que os pesos sejam atualizados, também define quantos padrões ler por vez e manter na memória
    # Number of Epochs: Número de vezes que o conjunto de dados de treinamento é mostrado à rede durante o treinamento
    batch_size = [1024]  # [10, 20, 40, 60, 80, 100, 1000]
    epochs = [1000]  #[10, 20, 50]

    # Training Optimization: Algoritmos de otimização de treinamento disponibilizados pelo Keras
    optimizer = ['Adam']  # ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

    # Learning Rate: A taxa de aprendizagem controla o quanto atualizar o peso no final de cada lote
    # Momentum: O momento controla o quanto deixar a atualização anterior influenciar a atualização do peso atual.
    learn_rate = [0.01, 0.1]  # [0.1, 0.2, 0.3]
    momentum = [0.0, 0.1]  # [0.2, 0.4, 0.6, 0.8, 0.9]

    # Network Weight Initialization: Inicialização do peso da rede neural
    init_mode = ['uniform']  # ['lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']  # ['uniform']

    # Neuron Activation Function: A função de ativação controla a não linearidade de neurônios individuais e quando disparar.
    activation = ['relu']  #['softmax', 'relu', 'softplus', 'softsign', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'] # ['softsign', 'softmax', 'relu']

    # Dropout Regularization:  Limitar o sobreajuste e melhorar a capacidade de generalização do modelo
    # Para obter bons resultados, é melhor combinado com uma restrição de peso, como a restrição de norma máxima.
    weight_constraint = [4]  # [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2]  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # [0.2]

    # Number of Neurons in the Hidden Layer: O número de neurônios em uma camada controla a capacidade de representação da rede
    neurons = [1, 10, 30]  # [1, 5, 10]

    param_grid = dict(  # optimizer=optimizer,
        batch_size=batch_size,
        epochs=epochs,
        learn_rate=learn_rate,
        momentum=momentum,
        init_mode=init_mode,
        activation=activation,
        #optimizer=optimizer
        weight_constraint=weight_constraint,
        dropout_rate=dropout_rate,
        neurons=neurons
    )

    return param_grid


def model_function_cnn(activation='relu', init_mode='uniform', neurons=1, dropout_rate=0.0, weight_constraint=0, learn_rate=0.01, momentum=0):
    model = Sequential()

    #model.add(Dense(neurons, activation='relu'))
    #model.add(Dense(neurons, activation='relu', kernel_constraint=max_norm(weight_constraint)))
    model.add(Dense(neurons, activation=activation, kernel_initializer=init_mode, kernel_constraint=max_norm(weight_constraint)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Número de saídas do classificador

    optimizer = SGD(lr=learn_rate, momentum=momentum)
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # print(model.summary())

    return model


def model_function_cnn_2(activation='relu', init_mode='uniform', neurons=1, dropout_rate=0.0, weight_constraint=0, learn_rate=0.01, momentum=0):
    model = Sequential()

    #model.add(Dense(neurons, activation='relu'))
    #model.add(Dense(neurons, activation='relu', kernel_constraint=max_norm(weight_constraint)))
    model.add(Dense(neurons, activation=activation, kernel_initializer=init_mode, kernel_constraint=max_norm(weight_constraint)))

    model.add(Conv2D(16, (5, 5), activation='relu', padding='same', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (4, 4), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Dropout(0.25))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Número de saídas do classificador

    optimizer = SGD(lr=learn_rate, momentum=momentum)
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # print(model.summary())

    return model


def model_AlexNet():
    # AlexNet expects input shape 227x227

    model = Sequential()

    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='softmax'))

    model.compile(optimizer=SGD(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    return model