import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import tensorflow_addons as tfa
from collections import deque

# function to crunch the entire video to 32 non empty frames and each frame only contains 92 indices (considering only lips in entire face)
class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PreprocessLayer, self).__init__()
        
    def pad_edge(self, t, repeats, side):
        if side == 'LEFT':
            return tf.concat((tf.repeat(t[:1], repeats=repeats, axis=0), t), axis=0)
        elif side == 'RIGHT':
            return tf.concat((t, tf.repeat(t[-1:], repeats=repeats, axis=0)), axis=0)
    
    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None,543,3], dtype=tf.float32),),
    )
    def call(self, data0):
        # Number of Frames in Video
        N_FRAMES0 = tf.shape(data0)[0]
        
        # Keep only non-empty frames in data
        frames_hands_nansum = tf.experimental.numpy.nanmean(tf.gather(data0, HAND_IDXS0, axis=1), axis=[1,2])
        non_empty_frames_idxs = tf.where(frames_hands_nansum > 0)
        non_empty_frames_idxs = tf.squeeze(non_empty_frames_idxs, axis=1)
        data = tf.gather(data0, non_empty_frames_idxs, axis=0)
        
        non_empty_frames_idxs = tf.cast(non_empty_frames_idxs, tf.float32) 
        
        # Number of non-empty frames
        N_FRAMES = tf.shape(data)[0]
        data = tf.gather(data, LANDMARK_IDXS0, axis=1)
        
        if N_FRAMES < 32:
            # Video fits in 32
            non_empty_frames_idxs = tf.pad(non_empty_frames_idxs, [[0, 32-N_FRAMES]], constant_values=-1)
            data = tf.pad(data, [[0, 32-N_FRAMES], [0,0], [0,0]], constant_values=0)
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            return data, non_empty_frames_idxs
        else:
            # Video needs to be downsampled to 32
            if N_FRAMES < 32**2:
                repeats = tf.math.floordiv(32 * 32, N_FRAMES0)
                data = tf.repeat(data, repeats=repeats, axis=0)
                non_empty_frames_idxs = tf.repeat(non_empty_frames_idxs, repeats=repeats, axis=0)

            # Pad To Multiple Of Input Size
            pool_size = tf.math.floordiv(len(data), 32)
            if tf.math.mod(len(data), 32) > 0:
                pool_size += 1
            if pool_size == 1:
                pad_size = (pool_size * 32) - len(data)
            else:
                pad_size = (pool_size * 32) % len(data)

            # Pad Start/End with Start/End value
            pad_left = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(32, 2)
            pad_right = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(32, 2)
            if tf.math.mod(pad_size, 2) > 0:
                pad_right += 1

            # Pad By Concatenating Left/Right Edge Values
            data = self.pad_edge(data, pad_left, 'LEFT')
            data = self.pad_edge(data, pad_right, 'RIGHT')

            # Pad Non Empty Frame Indices
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_left, 'LEFT')
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_right, 'RIGHT')

            # Reshape to Mean Pool
            data = tf.reshape(data, [32, -1, N_COLS, 3])
            non_empty_frames_idxs = tf.reshape(non_empty_frames_idxs, [32, -1])

            # Mean Pool
            data = tf.experimental.numpy.nanmean(data, axis=1)
            non_empty_frames_idxs = tf.experimental.numpy.nanmean(non_empty_frames_idxs, axis=1)

            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            
            return data, non_empty_frames_idxs

def scaled_dot_product(q,k,v, softmax, attention_mask):
    #calculates Q . K(transpose)
    qkt = tf.matmul(q,k,transpose_b=True)
    #caculates scaling factor
    dk = tf.math.sqrt(tf.cast(q.shape[-1],dtype=tf.float32))
    scaled_qkt = qkt/dk
    softmax = softmax(scaled_qkt, mask=attention_mask)
    
    z = tf.matmul(softmax,v)
    #shape: (m,Tx,depth), same shape as q,k,v
    return z

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,num_of_heads):
        super(MultiHeadAttention,self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model//num_of_heads
        self.wq = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wo = tf.keras.layers.Dense(d_model)
        self.softmax = tf.keras.layers.Softmax()
        
    def call(self,x, attention_mask):
        
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(scaled_dot_product(Q,K,V, self.softmax, attention_mask))
            
        multi_head = tf.concat(multi_attn,axis=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention

class Transformer(tf.keras.Model):
    def __init__(self, num_blocks):
        super(Transformer, self).__init__(name='transformer')
        self.num_blocks = num_blocks
    
    def build(self, input_shape):
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # First Layer Normalisation
            self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(UNITS, 8))
            # Second Layer Normalisation
            self.ln_2s.append(tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(UNITS * MLP_RATIO, activation=GELU, kernel_initializer=INIT_GLOROT_UNIFORM),
                tf.keras.layers.Dropout(MLP_DROPOUT_RATIO),
                tf.keras.layers.Dense(UNITS, kernel_initializer=INIT_HE_UNIFORM),
            ]))
        
    def call(self, x, attention_mask):
        # Iterate input over transformer blocks
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x1 = ln_1(x)
            attention_output = mha(x1, attention_mask)
            x2 = x1 + attention_output
            x3 = ln_2(x2)
            x3 = mlp(x3)
            x = x3 + x2
    
        return x
    
class LandmarkEmbedding(tf.keras.Model):
    def __init__(self, units, name):
        super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
        self.units = units
        
    def build(self, input_shape):
        # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = self.add_weight(
            name=f'{self.name}_empty_embedding',
            shape=[self.units],
            initializer=INIT_ZEROS,
        )
        # Embedding
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
        ], name=f'{self.name}_dense')

    def call(self, x):
        return tf.where(
                # Checks whether landmark is missing in frame
                tf.reduce_sum(x, axis=2, keepdims=True) == 0,
                # If so, the empty embedding is used
                self.empty_embedding,
                # Otherwise the landmark data is embedded
                self.dense(x),
            )

class CustomEmbedding(tf.keras.Model):
    def __init__(self):
        super(CustomEmbedding, self).__init__()
        
    def get_diffs(self, l):
        S = l.shape[2]
        other = tf.expand_dims(l, 3)
        other = tf.repeat(other, S, axis=3)
        other = tf.transpose(other, [0,1,3,2])
        diffs = tf.expand_dims(l, 3) - other
        diffs = tf.reshape(diffs, [-1, 32, S*S])
        return diffs

    def build(self, input_shape):
        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.keras.layers.Embedding(32+1, UNITS, embeddings_initializer=INIT_ZEROS)
        # Embedding layer for Landmarks
        self.lips_embedding = LandmarkEmbedding(LIPS_UNITS, 'lips')
        self.left_hand_embedding = LandmarkEmbedding(HANDS_UNITS, 'left_hand')
        self.right_hand_embedding = LandmarkEmbedding(HANDS_UNITS, 'right_hand')
        self.pose_embedding = LandmarkEmbedding(POSE_UNITS, 'pose')
        # Landmark Weights
        self.landmark_weights = tf.Variable(tf.zeros([4], dtype=tf.float32), name='landmark_weights')
        # Fully Connected Layers for combined landmarks
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(UNITS, name='fully_connected_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),
            tf.keras.layers.Dense(UNITS, name='fully_connected_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
        ], name='fc')


    def call(self, lips0, left_hand0, right_hand0, pose0, non_empty_frame_idxs, training=False):
        # Lips
        lips_embedding = self.lips_embedding(lips0)
        # Left Hand
        left_hand_embedding = self.left_hand_embedding(left_hand0)
        # Right Hand
        right_hand_embedding = self.right_hand_embedding(right_hand0)
        # Pose
        pose_embedding = self.pose_embedding(pose0)
        # Merge Embeddings of all landmarks with mean pooling
        x = tf.stack((lips_embedding, left_hand_embedding, right_hand_embedding, pose_embedding), axis=3)
        # Merge Landmarks with trainable attention weights
        x = x * tf.nn.softmax(self.landmark_weights)
        x = tf.reduce_sum(x, axis=3)
        # Fully Connected Layers
        x = self.fc(x)
        # Add Positional Embedding
        normalised_non_empty_frame_idxs = tf.where(
            tf.math.equal(non_empty_frame_idxs, -1.0),
            32,
            tf.cast(
                non_empty_frame_idxs / tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True) * 32,
                tf.int32,
            ),
        )
        x = x + self.positional_embedding(normalised_non_empty_frame_idxs)
        
        return x

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def get_model():
    # Inputs
    frames = tf.keras.layers.Input([32, N_COLS, 3], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([32], dtype=tf.float32, name='non_empty_frame_idxs')
    # Padding Mask
    mask = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
    mask = tf.expand_dims(mask, axis=2)
    
    x = frames
    x = tf.slice(x, [0,0,0,0], [-1,32, N_COLS, 2])
    # LIPS
    lips = tf.slice(x, [0,0,LIPS_START,0], [-1,32, 40, 2])
    lips = tf.where(
            tf.math.equal(lips, 0.0),
            0.0,
            (lips - LIPS_MEAN) / LIPS_STD,
        )
    lips = tf.reshape(lips, [-1, 32, 40*2])
    # LEFT HAND
    left_hand = tf.slice(x, [0,0,40,0], [-1,32, 21, 2])
    left_hand = tf.where(
            tf.math.equal(left_hand, 0.0),
            0.0,
            (left_hand - LEFT_HANDS_MEAN) / LEFT_HANDS_STD,
        )
    left_hand = tf.reshape(left_hand, [-1, 32, 21*2])
    # RIGHT HAND
    right_hand = tf.slice(x, [0,0,61,0], [-1,32, 21, 2])
    right_hand = tf.where(
            tf.math.equal(right_hand, 0.0),
            0.0,
            (right_hand - RIGHT_HANDS_MEAN) / RIGHT_HANDS_STD,
        )
    right_hand = tf.reshape(right_hand, [-1, 32, 21*2])
    # POSE
    pose = tf.slice(x, [0,0,82,0], [-1,32, 10, 2])
    pose = tf.where(
            tf.math.equal(pose, 0.0),
            0.0,
            (pose - POSE_MEAN) / POSE_STD,
        )
    pose = tf.reshape(pose, [-1, 32, 10*2])
    x = lips, left_hand, right_hand, pose
    x = CustomEmbedding()(lips, left_hand, right_hand, pose, non_empty_frame_idxs)
    # Encoder Transformer Blocks
    x = Transformer(NUM_BLOCKS)(x, mask)
    # Pooling
    x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    # Classification Layer
    x = tf.keras.layers.Dense(250, activation=tf.keras.activations.softmax, kernel_initializer=INIT_GLOROT_UNIFORM)(x)
    outputs = x
    
    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    
    # Simple Categorical Crossentropy Loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Adam Optimizer with weight decay
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)
    
    lr_metric = get_lr_metric(optimizer)
    metrics = ["acc",lr_metric]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def load_relevant_data_subset(data):
    data_columns = ['x', 'y', 'z']
    data = data[data_columns]
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def get_data(data_frame):
    data = load_relevant_data_subset(data_frame)
    data = preprocess_layer(data)
    return data

def predict(data_frame , index_to_value , model_one):
    data, non_empty_frames_idxs = get_data(data_frame)
    data = np.array(data).reshape(1,32,92,3)
    non_empty_frames_idxs = np.array(non_empty_frames_idxs).reshape(1,32)
    try:
        y_val_pred = model_one.predict({ 'frames': data, 'non_empty_frame_idxs': non_empty_frames_idxs } , verbose = 0)
    except Exception as e:
        return "No gesture detected!"
    confidence = y_val_pred[0][y_val_pred.argmax(axis = 1)[0]]
    if confidence == None or confidence < 0.6:
        return "No gesture detected!"
    
    return (index_to_value[y_val_pred.argmax(axis = 1)[0]] , confidence)

def detect_gesture(image , holistic):
    x = []
    y = []
    z = []
    results = holistic.process(image)
    #face
    if(results.face_landmarks is None):
        for i in range(468):
            x.append(None)
            y.append(None)
            z.append(None)
    else:
        for ind,val in enumerate(results.face_landmarks.landmark):
            x.append(val.x)
            y.append(val.y)
            z.append(val.z)
    #left hand
    if(results.left_hand_landmarks is None):
        for i in range(21):
            x.append(None)
            y.append(None)
            z.append(None)
    else:
        for ind,val in enumerate(results.left_hand_landmarks.landmark):
            x.append(val.x)
            y.append(val.y)
            z.append(val.z)
    #pose
    if(results.pose_landmarks is None):
        for i in range(33):
            x.append(None)
            y.append(None)
            z.append(None)
    else:
        for ind,val in enumerate(results.pose_landmarks.landmark):
            x.append(val.x)
            y.append(val.y)
            z.append(val.z)
    #right hand
    if(results.right_hand_landmarks is None):
        for i in range(21):
            x.append(None)
            y.append(None)
            z.append(None)
    else:
        for ind,val in enumerate(results.right_hand_landmarks.landmark):
            x.append(val.x)
            y.append(val.y)
            z.append(val.z)

    return pd.DataFrame({
        "x" : x,
        "y" : y,
        "z" : z
    })

def main():
    try:
        dq = deque()

        model_one = get_model()
        
        model_one.load_weights('assets/final_model_one_v1/final_model_one_weights')

        with open('assets/sign_to_prediction_index_map.json', 'r') as f:
            data = json.load(f)

        index_to_value = {value: str(index) for index, value in data.items()}

        frame_number = 0
        mp_holistic = mp.solutions.holistic

        cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():

                success, image = cap.read()
                if not success:
                    break

                frame_number += 1

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = detect_gesture(image, holistic)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('MediaPipe Holistic', image)
                
                dq.append(results)
                # print(len(dq))
                if frame_number == 20:
                    data_frame = pd.concat(dq)
                    print(predict(data_frame , index_to_value , model_one))
                    dq.clear()
                    frame_number = 0

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(e)

if __name__ == '__main__':

    ROWS_PER_FRAME = 543

    # Epsilon value for layer normalisation
    LAYER_NORM_EPS = 1e-6

    # Dense layer units for landmarks
    LIPS_UNITS = 384
    HANDS_UNITS = 384
    POSE_UNITS = 384
    # final embedding and transformer embedding size
    UNITS = 384

    # Transformer
    NUM_BLOCKS = 2
    MLP_RATIO = 2

    # Dropout
    EMBEDDING_DROPOUT = 0.00
    MLP_DROPOUT_RATIO = 0.30
    CLASSIFIER_DROPOUT_RATIO = 0.10

    # Initiailizers
    INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
    INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
    INIT_ZEROS = tf.keras.initializers.constant(0.0)
    # Activations
    GELU = tf.keras.activations.gelu

    # landmark indices in original data
    LIPS_IDXS0 = np.array([
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ])

    LEFT_HAND_IDXS0  = np.arange(468,489)
    RIGHT_HAND_IDXS0 = np.arange(522,543)
    POSE_IDXS0       = np.arange(502, 512)
    LANDMARK_IDXS0   = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, POSE_IDXS0))
    HAND_IDXS0       = np.concatenate((LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0), axis=0)
    N_COLS           = LANDMARK_IDXS0.size

    # Landmark indices in processed data
    LIPS_IDXS       = np.argwhere(np.isin(LANDMARK_IDXS0, LIPS_IDXS0)).squeeze()
    LEFT_HAND_IDXS  = np.argwhere(np.isin(LANDMARK_IDXS0, LEFT_HAND_IDXS0)).squeeze()
    RIGHT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, RIGHT_HAND_IDXS0)).squeeze()
    HAND_IDXS       = np.argwhere(np.isin(LANDMARK_IDXS0, HAND_IDXS0)).squeeze()
    POSE_IDXS       = np.argwhere(np.isin(LANDMARK_IDXS0, POSE_IDXS0)).squeeze()

    LIPS_START = 0
    LEFT_HAND_START = LIPS_IDXS.size
    RIGHT_HAND_START = LEFT_HAND_START + LEFT_HAND_IDXS.size
    POSE_START = RIGHT_HAND_START + RIGHT_HAND_IDXS.size

    LIPS_MEAN = np.load('assets/lips_mean.npy')
    LIPS_STD = np.load('assets/lips_std.npy')

    LEFT_HANDS_MEAN = np.load('assets/lh_mean.npy')
    LEFT_HANDS_STD = np.load('assets/lh_std.npy')
    RIGHT_HANDS_MEAN = np.load('assets/rh_mean.npy')
    RIGHT_HANDS_STD = np.load('assets/rh_std.npy')

    POSE_MEAN = np.load('assets/pose_mean.npy')
    POSE_STD = np.load('assets/pose_std.npy')

    preprocess_layer = PreprocessLayer()
    
    main()