import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.learning.models import keras_utils
# Corrected import from the specific data preparation file for this task
from data_preparation_p3_arbitrator import NUM_P3_ARB_FEATURES as IMPORTED_NUM_FEATURES, ELEMENT_SPEC_P3_ARB as IMPORTED_ELEMENT_SPEC

# Define task-specific variables as per PDF
NUM_P3_TASK_FEATURES = IMPORTED_NUM_FEATURES
ELEMENT_SPEC_P3_TASK = IMPORTED_ELEMENT_SPEC

# Define the batched input spec for TFF's from_keras_model
BATCHED_INPUT_SPEC = (
    tf.TensorSpec(shape=(None, NUM_P3_TASK_FEATURES), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
)

def create_keras_model():
    """Creates an uncompiled Keras model with batch normalization for stability."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_P3_TASK_FEATURES,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Debug prints
    print("\nModel summary:")
    model.summary()
    
    # Test with sample input
    sample_input = tf.random.normal([1, NUM_P3_TASK_FEATURES])
    print(f"\nSample input shape: {sample_input.shape}")
    sample_output = model(sample_input)
    print(f"Sample output shape: {sample_output.shape}")
    
    return model

def model_fn():
    """Creates a TFF model from an uncompiled Keras model."""
    keras_model = create_keras_model()
    tff_model = keras_utils.from_keras_model(
        keras_model,
        input_spec=BATCHED_INPUT_SPEC,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC()
        ]
    )
    
    # Debug prints
    print("\nTFF model input spec:")
    print(tff_model.input_spec)
    
    return tff_model

if __name__ == '__main__':
    # Test model creation and TFF wrapping
    print("Testing Keras model creation...")
    keras_m = create_keras_model()
    keras_m.summary()
    
    # Try with a sample 2D input (batch of N)
    sample_input_2d_batchN = tf.random.normal([5, NUM_P3_TASK_FEATURES])
    print(f"\nShape of 2D (batch N) sample input: {sample_input_2d_batchN.shape}")
    try:
        output_2d_batchN = keras_m(sample_input_2d_batchN)
        print(f"Output shape with 2D (batch N) input: {output_2d_batchN.shape}")
    except Exception as e:
        print(f"Error with 2D (batch N) input: {e}")

    print("\nTesting TFF model function...")
    try:
        tff_m_instance = model_fn()
        print(f"TFF model function created successfully. Input spec: {tff_m_instance.input_spec}")
    except Exception as e:
        print(f"Error creating TFF model instance: {e}")

