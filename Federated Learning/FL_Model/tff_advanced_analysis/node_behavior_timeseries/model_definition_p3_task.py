import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.learning.models import keras_utils
# Corrected import from the specific data preparation file for this task
from data_preparation_p3_timeseries import (
    NUM_P3_TS_FEATURES as IMPORTED_NUM_FEATURES, 
    ELEMENT_SPEC_P3_TS as IMPORTED_ELEMENT_SPEC,
    TIMESTEPS as IMPORTED_TIMESTEPS # Import TIMESTEPS
)

# Define task-specific variables as per PDF
NUM_P3_TASK_FEATURES = IMPORTED_NUM_FEATURES
TIMESTEPS_P3_TASK = IMPORTED_TIMESTEPS # Use the imported TIMESTEPS
# ELEMENT_SPEC_P3_TASK = IMPORTED_ELEMENT_SPEC # This is the unbatched spec

# Define the batched input spec for TFF's from_keras_model
BATCHED_INPUT_SPEC_P3_TASK = (
    tf.TensorSpec(shape=(None, TIMESTEPS_P3_TASK, NUM_P3_TASK_FEATURES), dtype=tf.float32),
    tf.TensorSpec(shape=(None, TIMESTEPS_P3_TASK, NUM_P3_TASK_FEATURES), dtype=tf.float32)
)

def create_keras_model():
    """Creates an uncompiled Keras autoencoder model for time-series anomaly detection."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(TIMESTEPS_P3_TASK, NUM_P3_TASK_FEATURES)),
        tf.keras.layers.LSTM(128, activation='relu', return_sequences=False), # Encoder
        tf.keras.layers.RepeatVector(TIMESTEPS_P3_TASK),  # Use TIMESTEPS_P3_TASK here
        tf.keras.layers.LSTM(128, activation='relu', return_sequences=True), # Decoder
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(NUM_P3_TASK_FEATURES)) # Output layer
    ])
    print("\nTime-Series Model summary:")
    model.summary(print_fn=print)
    return model

def model_fn():
    """Creates a TFF model from an uncompiled Keras model."""
    keras_model = create_keras_model()
    tff_model = keras_utils.from_keras_model(
        keras_model,
        input_spec=BATCHED_INPUT_SPEC_P3_TASK, # Use the BATCHED input spec here
        loss=tf.keras.losses.MeanSquaredError(), # Reconstruction loss for autoencoder
        metrics=[
            tf.keras.metrics.MeanSquaredError() # Track reconstruction error
        ]
    )
    print("\nTime-Series TFF model input spec (should be batched):")
    print(tff_model.input_spec)
    return tff_model

if __name__ == '__main__':
    print("Testing Keras model creation for Time-Series Anomaly Detection...")
    keras_m = create_keras_model()
    
    sample_input_2d_batchN = tf.random.normal([5, TIMESTEPS_P3_TASK, NUM_P3_TASK_FEATURES])
    print(f"\nShape of 2D (batch N) sample input for Time-Series: {sample_input_2d_batchN.shape}")
    try:
        output_2d_batchN = keras_m(sample_input_2d_batchN)
        print(f"Output shape with 2D (batch N) input for Time-Series: {output_2d_batchN.shape}")
    except Exception as e:
        print(f"Error with 2D (batch N) input for Time-Series: {e}")

    print("\nTesting TFF model function for Time-Series Anomaly Detection...")
    try:
        tff_m_instance = model_fn()
        print(f"Time-Series TFF model function created successfully. Input spec: {tff_m_instance.input_spec}")
    except Exception as e:
        print(f"Error creating Time-Series TFF model instance: {e}")

