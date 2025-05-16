import tensorflow as tf
import tensorflow_federated as tff
from data_preparation_phase2 import NUM_PHASE2_FEATURES # ELEMENT_SPEC_PHASE2 is not needed here directly

# Define the batched input spec for TFF's from_keras_model
# This describes the shape of one batch of data (features, labels)
BATCHED_INPUT_SPEC = (
    tf.TensorSpec(shape=(None, NUM_PHASE2_FEATURES), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
)

def create_keras_model():
    """Creates a Keras model for binary classification of anomalous/collusive behavior."""
    model = tf.keras.models.Sequential([
        # Input layer with explicit shape for Phase 2 features (shape of one sample)
        tf.keras.layers.Input(shape=(NUM_PHASE2_FEATURES,)),
        
        # First hidden layer with increased capacity
        tf.keras.layers.Dense(
            units=32,  # Increased from 16 to handle more complex patterns
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            kernel_regularizer=tf.keras.regularizers.l2(0.01)  # Added L2 regularization
        ),
        tf.keras.layers.BatchNormalization(),  # Added batch normalization
        tf.keras.layers.Dropout(0.3),
        
        # Second hidden layer
        tf.keras.layers.Dense(
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer for binary classification
        tf.keras.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid
        )
    ])
    return model

def tff_model_fn():
    """Wraps the Keras model for use with TFF.
    Returns a tff.learning.models.VariableModel.
    """
    keras_model_instance = create_keras_model()
    
    return tff.learning.models.from_keras_model(
        keras_model_instance,
        input_spec=BATCHED_INPUT_SPEC,  # Use the defined batched input spec
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),  # Added precision metric
            tf.keras.metrics.Recall(name="recall")  # Added recall metric
        ]
    )

if __name__ == '__main__':
    # Test model creation and TFF wrapping
    print("Testing Keras model creation...")
    keras_m = create_keras_model()
    keras_m.summary()
    
    # Try with a sample 2D input (batch of N)
    sample_input_2d_batchN = tf.random.normal([5, NUM_PHASE2_FEATURES])
    print(f"\nShape of 2D (batch N) sample input: {sample_input_2d_batchN.shape}")
    try:
        output_2d_batchN = keras_m(sample_input_2d_batchN)
        print(f"Output shape with 2D (batch N) input: {output_2d_batchN.shape}")
    except Exception as e:
        print(f"Error with 2D (batch N) input: {e}")

    print("\nTesting TFF model function...")
    # The tff_model_fn itself doesn't take arguments for its direct call here for testing purposes.
    # It's used by TFF processes which will handle data according to the input_spec.
    try:
        tff_m_instance = tff_model_fn()
        print(f"TFF model function created successfully. Input spec: {tff_m_instance.input_spec}")
    except Exception as e:
        print(f"Error creating TFF model instance: {e}")

