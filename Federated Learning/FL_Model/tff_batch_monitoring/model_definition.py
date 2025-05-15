import tensorflow as tf
import tensorflow_federated as tff
from data_preparation_phase2 import NUM_PHASE2_FEATURES, ELEMENT_SPEC_PHASE2

def create_keras_model():
    """Creates a Keras model for binary classification of anomalous/collusive behavior."""
    model = tf.keras.models.Sequential([
        # Input layer with explicit shape for Phase 2 features
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
        input_spec=ELEMENT_SPEC_PHASE2,  # Using Phase 2 element spec
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
    
    print("\nTesting TFF model function...")
    tff_m_fn = tff_model_fn()
    print("TFF model function created successfully.")
    
    # Example: Create a concrete TFF model (not usually done directly like this for training)
    # state_manager = tff.learning.models.ModelWeights.get_model_weights(tff_m_fn())
    # print(f"Model weights structure: {state_manager}")