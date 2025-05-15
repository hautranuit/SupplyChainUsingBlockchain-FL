import tensorflow as tf
import tensorflow_federated as tff
from data_preparation import NUM_FEATURES, ELEMENT_SPEC

def create_keras_model():
    """Creates a simple Keras model for binary classification."""
    model = tf.keras.models.Sequential([
        # Input layer with explicit shape
        tf.keras.layers.Input(shape=(NUM_FEATURES,)),
        tf.keras.layers.Dense(
            units=16,  # Increased units
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),  # Good default
        ),
        tf.keras.layers.Dropout(0.3),  # Added dropout for regularization
        tf.keras.layers.Dense(
            units=8,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        ),
        tf.keras.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid  # Sigmoid for binary classification probability
        )
    ])
    return model

def tff_model_fn():
    """Wraps the Keras model for use with TFF.
    Returns a tff.learning.models.VariableModel.
    """
    keras_model_instance = create_keras_model()
    
    # Create input spec that exactly matches the dataset element shapes
    # The dataset has elements of type <float32[?,5],int32[?,1]>
    single_example_spec = (
        tf.TensorSpec(shape=(None, NUM_FEATURES), dtype=tf.float32),  # features with variable batch size
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32)  # labels with shape (batch_size, 1)
    )
    
    return tff.learning.models.from_keras_model(
        keras_model_instance,
        input_spec=single_example_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

if __name__ == '__main__':
    # Test model creation and TFF wrapping
    print("Testing Keras model creation...")
    keras_m = create_keras_model()
    keras_m.summary()
    
    print("\nTesting TFF model function...")
    tff_m_fn = tff_model_fn()
    print("TFF model function created successfully.")
    
    # Example: Create a concrete TFF model (not usually done directly like this for training)
    # state_manager = tff.learning.models.ModelWeights.get_model_weights(tff_m_fn())
    # print(f"Model weights structure: {state_manager}")