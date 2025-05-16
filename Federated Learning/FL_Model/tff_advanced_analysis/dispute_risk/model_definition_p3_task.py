import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.learning.models import keras_utils
# Corrected import from the specific data preparation file for this task
from data_preparation_p3_dispute import NUM_P3_DIS_FEATURES as IMPORTED_NUM_FEATURES, ELEMENT_SPEC_P3_DIS as IMPORTED_ELEMENT_SPEC

# Define task-specific variables as per PDF
NUM_P3_TASK_FEATURES = IMPORTED_NUM_FEATURES
# ELEMENT_SPEC_P3_TASK = IMPORTED_ELEMENT_SPEC # This is the unbatched spec

# Define the batched input spec for TFF's from_keras_model, similar to arbitrator_bias task
BATCHED_INPUT_SPEC_P3_TASK = (
    tf.TensorSpec(shape=(None, NUM_P3_TASK_FEATURES), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 1), dtype=tf.int32) # Matches ELEMENT_SPEC_P3_DIS label dtype
)

def create_keras_model():
    """Creates an uncompiled Keras model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(NUM_P3_TASK_FEATURES,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # Debug prints for model structure (optional, can be removed for final version)
    print("\nDispute Risk Model summary:")
    model.summary(print_fn=print)
    sample_input = tf.random.normal([1, NUM_P3_TASK_FEATURES])
    print(f"Dispute Risk Sample input shape: {sample_input.shape}")
    sample_output = model(sample_input)
    print(f"Dispute Risk Sample output shape: {sample_output.shape}")
    return model

def model_fn():
    """Creates a TFF model from an uncompiled Keras model."""
    keras_model = create_keras_model()
    tff_model = keras_utils.from_keras_model(
        keras_model,
        input_spec=BATCHED_INPUT_SPEC_P3_TASK, # Use the BATCHED input spec here
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC()
        ]
    )
    print("\nDispute Risk TFF model input spec (should be batched):")
    print(tff_model.input_spec)
    return tff_model

if __name__ == '__main__':
    print("Testing Keras model creation for Dispute Risk...")
    keras_m = create_keras_model()
    # keras_m.summary() # Already printed in create_keras_model
    
    sample_input_2d_batchN = tf.random.normal([5, NUM_P3_TASK_FEATURES])
    print(f"\nShape of 2D (batch N) sample input for Dispute Risk: {sample_input_2d_batchN.shape}")
    try:
        output_2d_batchN = keras_m(sample_input_2d_batchN)
        print(f"Output shape with 2D (batch N) input for Dispute Risk: {output_2d_batchN.shape}")
    except Exception as e:
        print(f"Error with 2D (batch N) input for Dispute Risk: {e}")

    print("\nTesting TFF model function for Dispute Risk...")
    try:
        tff_m_instance = model_fn()
        print(f"Dispute Risk TFF model function created successfully. Input spec: {tff_m_instance.input_spec}")
    except Exception as e:
        print(f"Error creating Dispute Risk TFF model instance: {e}")

