from tensorflow.contrib.learn import Experiment
import tensorflow as tf

from pitycnn.inputs import Inputs
from pitycnn.model import model_fn


def generate_experiment_fn(train_images, train_labels, valid_images, valid_labels, n_classes, batch_size = 10):
    def experiment_fn(run_config, params):
        run_config = run_config.replace(save_checkpoints_steps=params.min_eval_frequency)
        estimator = get_estimator(run_config, params)

        train_inputs = Inputs(train_images, train_labels, n_classes, batch_size=batch_size, shuffle=True)
        valid_inputs = Inputs(valid_images, valid_labels, n_classes, name="valid", batch_size=batch_size)

        with tf.device('/cpu:0'):
            train_input_fn, train_input_hook = train_inputs.generate_input_fn()
            eval_input_fn, eval_input_hook = valid_inputs.generate_input_fn()

        experiment = Experiment(
            estimator=estimator,  # Estimator
            train_input_fn=train_input_fn,  # First-class function
            eval_input_fn=eval_input_fn,  # First-class function
            train_steps=params.train_steps,  # Minibatch steps
            min_eval_frequency=params.min_eval_frequency,  # Eval frequency
            train_monitors=[train_input_hook],  # Hooks for training
            eval_hooks=[eval_input_hook],  # Hooks for evaluation
            eval_steps=None  # Use evaluation feeder until its empty
        )

        return experiment

    def get_estimator(run_config, params):
        print(run_config)
        return tf.estimator.Estimator(
            model_fn=model_fn,  # First-class function
            params=params,  # HParams
            config=run_config  # RunConfig
        )

    return experiment_fn
