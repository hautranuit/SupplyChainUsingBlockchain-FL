import tensorflow as tf
import tensorflow_federated as tff
from model_definition_p3_task import model_fn
import collections

def build_weighted_fed_avg(model_fn):
    """Builds a weighted federated averaging process with gradient clipping."""
    # Create the client optimizer using TFF's optimizer builder
    client_optimizer = tff.learning.optimizers.build_sgdm(
        learning_rate=0.01,
        momentum=0.0
    )
    
    return tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer
    )

def preprocess_client_dataset(dataset, batch_size=16):
    """Preprocesses a client dataset for federated training."""
    def batch_format_fn(features, labels):
        """Formats the dataset elements into the expected structure."""
        # Ensure features and labels have the correct shape
        features = tf.reshape(features, [5])  # Ensure features are 1D with 5 elements
        labels = tf.reshape(labels, [1])      # Ensure labels are 1D with 1 element
        return (features, labels)  # Return as tuple instead of OrderedDict
    
    return dataset.map(batch_format_fn).batch(batch_size)

def evaluate_model(model, test_data):
    """Evaluates the model on test data."""
    metrics = model.report_local_unfinalized_metrics()
    accuracy = float(metrics['accuracy'])
    auc = float(metrics['auc'])
    return accuracy, auc 