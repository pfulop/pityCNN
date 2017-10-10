from tensorflow.contrib.learn.python.learn import learn_runner


def main(train_files, eval_files, batch_size=100, num_epochs=1000, learning_rate=0.01,  job_dir='model'):
    experiment_fn = generate_experiment_fn(
        train_files=train_files,
        eval_files=eval_files,
        batch_size=batch_size,
        num_epochs=num_epochs)

    params = tf.contrib.training.HParams(
        learning_rate=learning_rate,
        n_classes=10,
        train_steps=5000,
        min_eval_frequency=100
    )

    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=job_dir)

    learn_runner.run(experiment_fn, run_config=run_config, schedule="train_and_evaluate", hparams=params)


if __name__ == '__main__':
    main()
