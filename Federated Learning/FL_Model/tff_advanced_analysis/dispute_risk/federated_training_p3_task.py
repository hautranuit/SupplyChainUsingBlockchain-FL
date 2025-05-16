import tensorflow as tf
import tensorflow_federated as tff
from model_definition_p3_task import model_fn

def build_weighted_fed_avg():
    """Builds a weighted federated averaging process."""
    return tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.01) # Adjusted learning rate for consistency
    )

def preprocess_client_dataset(dataset, batch_size=32):
    """Preprocesses a client dataset for training."""
    def batch_format_fn(features, labels):
        """Formats the dataset elements into the expected structure."""
        # Ensure features and labels have the correct shape for this task (6 features)
        features = tf.reshape(features, [6])  # NUM_P3_DIS_FEATURES is 6
        labels = tf.reshape(labels, [1])      # Ensure labels are 1D with 1 element
        return (features, labels) # Return as tuple
    
    # Apply the mapping function, then shuffle, batch, and prefetch
    return dataset.map(batch_format_fn).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def evaluate_model(model, test_data):
    """Evaluates the model on test data."""
    # This function might not be directly used by run_simulation_p3_task.py if metrics are taken from fed_avg.next()
    # However, if it were used, it would need to be compatible with how TFF models report metrics.
    # For simplicity, the main simulation script extracts metrics directly from the federated process.
    # Placeholder if direct evaluation was needed:
    # metrics = model.evaluate(test_data, return_dict=True) 
    # return metrics.get('accuracy', float('nan')), metrics.get('auc', float('nan'))
    # The original code was: metrics = model.report_local_unfinalized_metrics()
    # This is for TFF models. If 'model' is a Keras model, it's different.
    # Assuming 'model' here refers to a TFF model state or similar context where this is valid.
    # However, the run_simulation script gets metrics from `fed_avg.next()`
    pass # This function is not directly called by the run_simulation script in the provided context.

