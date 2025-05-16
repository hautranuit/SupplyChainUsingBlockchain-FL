import tensorflow as tf
import tensorflow_federated as tff
from model_definition_p3_task import model_fn

def build_weighted_fed_avg():
    """Builds a weighted federated averaging process."""
    return tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.1)
    )

def preprocess_client_dataset(dataset, batch_size=32):
    """Preprocesses a client dataset for training."""
    return dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def evaluate_model(model, test_data):
    """Evaluates the model on test data."""
    metrics = model.report_local_unfinalized_metrics()
    return {
        'reconstruction_loss': float(metrics['mean_squared_error'].result())
    }

def detect_anomalies(model, data, threshold=0.1):
    """Detects anomalies in the data based on reconstruction error."""
    predictions = model.predict(data)
    reconstruction_errors = tf.reduce_mean(
        tf.square(data - predictions),
        axis=[1, 2]  # Average over timesteps and features
    )
    return reconstruction_errors > threshold 