import numpy as np

def gradient_descent_step(values, gradients, learning_rate):
    """
    Returns: updated values and the predicted first-order objective change
    """
    values = np.asarray(values, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)
    learning_rate = np.float64(learning_rate)

    delta = -learning_rate * gradients
    
    updated_values = values + delta
    
    predicted_change = delta.dot(gradients)
    
    return ([float(x) for x in updated_values], float(predicted_change))