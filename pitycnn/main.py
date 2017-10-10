from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn import RunConfig
from tensorflow.contrib.training import HParams

from pitycnn.experiment import generate_experiment_fn
from pitycnn.prep import prepare_data


def main(files, gpu_memory_fraction=0.8, min_eval_frequency=500, train_steps=5000, n_classes=10,
         learning_rate=0.01,
         job_dir='model'):
    params = HParams(
        learning_rate=learning_rate,
        n_classes=n_classes,
        train_steps=train_steps,
        min_eval_frequency=min_eval_frequency
    )

    train_images, train_labels, valid_images, valid_labels, n_classes = prepare_data(files)

    experiment_fn = generate_experiment_fn(train_images, train_labels, valid_images, valid_labels, n_classes)

    run_config = RunConfig(gpu_memory_fraction=gpu_memory_fraction)
    run_config = run_config.replace(model_dir=job_dir)

    learn_runner.run(experiment_fn, run_config=run_config, schedule="train_and_evaluate", hparams=params)
