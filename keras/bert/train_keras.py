import argparse
import logging
import os
import sys
import json
import time

import tensorflow as tf
import importlib.util

from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

import smdistributed.dataparallel.tensorflow.keras as dist
#import horovod.keras as dist

# initial distributed training
dist.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], "GPU")


def main():

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--train_batch_size", type=int, default=12)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--model_name_or_path", type=str, default='bert-large-uncased')
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_eval", type=bool, default=False)

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(args)

    #
    # Preprocesssing
    #

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load dataset
    train_dataset, test_dataset = load_dataset("imdb", split=["train", "test"])

    # Preprocess train dataset
    train_dataset = train_dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="longest"), batched=True
    )
    train_dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "label"])

    train_features = {
        x: train_dataset[x] for x in ["input_ids", "attention_mask"]
    }
    tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_dataset["label"])).batch(
        args.train_batch_size
    )

    # Preprocess test dataset
    test_dataset = test_dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length"), batched=True
    )
    test_dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "label"])

    test_features = {
        x: test_dataset[x] for x in ["input_ids", "attention_mask"]
    }
    tf_test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_dataset["label"])).batch(
        args.eval_batch_size
    )

    #
    # Training
    #

    # adjust optimizer
    # https://sagemaker.readthedocs.io/en/stable/api/training/sdp_versions/smd_data_parallel_tensorflow.html#smdistributed.dataparallel.tensorflow.DistributedOptimizer
    learning_rate = args.learning_rate * dist.size()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = dist.DistributedOptimizer(optimizer)

    # fine optimizer and loss
    model = TFAutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    # Specify `experimental_run_tf_function=False` to ensure TensorFlow uses dist.DistributedOptimizer() to compute gradients.
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, experimental_run_tf_function=False)

    # callbacks
    # https://horovod.readthedocs.io/en/stable/api.html#horovod.tensorflow.keras.callbacks.BroadcastGlobalVariablesCallback
    BroadcastGlobalVariablesCallback = dist.callbacks.BroadcastGlobalVariablesCallback

    callbacks = [
        # broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        BroadcastGlobalVariablesCallback(0),
    ]

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        start_time = time.time()
        # batch_size https://github.com/horovod/horovod/issues/1617
        # will be spread across all devices equally so. E.g.: train_batch_size 8, n_gpus 4 === 32
        train_results = model.fit(
            tf_train_dataset,
            epochs=1,
            steps_per_epoch=100,
            callbacks=callbacks,
            validation_batch_size=args.eval_batch_size,
            batch_size=args.train_batch_size,
            verbose=1 if dist.rank() == 0 else 0,
        )
        train_runtime = {f"train_runtime": round(time.time() - start_time, 4)}
        logger.info(f"train_runtime = {train_runtime}\n")

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        result = model.evaluate(tf_test_dataset, batch_size=args.eval_batch_size, return_dict=True)

if __name__ == "__main__":
    main()
