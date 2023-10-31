import json
import time
from pirlib.iotypes import DirectoryPath, FilePath
from pirlib.pipeline import pipeline
from pirlib.task import task
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from .utils import tuning, distillation, load_hyperparameters, inference

num_samples = 100


@task(cache=True, cache_key_file="hparams", timer=True)
def data_preprocessing(dataset: DirectoryPath, *, hparams: FilePath) -> DirectoryPath:
    """Data Preprocessing Stage."""
    start_time = time.time()
    hyperparameters_dict = load_hyperparameters(dataset)
    train_inputs_encodings, train_summaries_encodings = torch.load(
        dataset / "tokenized_train_data.pt"
    )
    val_inputs_encodings, val_summaries_encodings = torch.load(
        dataset / "tokenized_validation_data.pt"
    )

    if num_samples > 0:
        train_inputs_encodings = {
            key: value[:num_samples] for key, value in train_inputs_encodings.items()
        }
        train_summaries_encodings = {
            key: value[:num_samples] for key, value in train_summaries_encodings.items()
        }
        val_inputs_encodings = {
            key: value[:num_samples] for key, value in val_inputs_encodings.items()
        }
        val_summaries_encodings = {
            key: value[:num_samples] for key, value in val_summaries_encodings.items()
        }

    train_dataset = TensorDataset(
        train_inputs_encodings["input_ids"],
        train_inputs_encodings["attention_mask"],
        train_summaries_encodings["input_ids"],
        train_summaries_encodings["attention_mask"],
    )

    val_dataset = TensorDataset(
        val_inputs_encodings["input_ids"],
        val_inputs_encodings["attention_mask"],
        val_summaries_encodings["input_ids"],
        val_summaries_encodings["attention_mask"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        hyperparameters_dict["0__batch_size"],
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        hyperparameters_dict["0__batch_size"],
        shuffle=False,
    )

    original_data = load_dataset("cnn_dailymail", "3.0.0")
    test_dataset = original_data["test"]
    test_dataset = test_dataset.shuffle(seed=42).select([i for i in range(10)])

    output_dir = task.context().output

    # Write the data in the output directory.
    test_dataset.save_to_disk(output_dir / "testing_data")
    torch.save(train_dataloader, output_dir / "training_data")
    torch.save(val_dataloader, output_dir / "validation_data")
    with (output_dir / "data_preprocessing.txt").open("w") as f:
        f.write("Data preprocessing has been completed!")

    end_time = time.time()
    metrics = {"cost": end_time - start_time}
    with open(output_dir / "metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file)

    return output_dir


@task(cache=True, cache_key_file="hparams", timer=True)
def fine_tuning(
    data_path: DirectoryPath, dataset: DirectoryPath, *, hparams: FilePath
) -> DirectoryPath:
    """Fine Tuning Stage."""
    start_time = time.time()
    model_name = "t5-small"
    hyperparameters_dict = load_hyperparameters(dataset)

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda:0")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hyperparameters_dict["1__learning_rate"]
    )

    train_dataloader = torch.load(data_path / "training_data")
    val_dataloader = torch.load(data_path / "validation_data")
    validation_dataset = datasets.load_from_disk(dataset / "validation_data")
    if num_samples > 0 and num_samples < 13368:
        validation_dataset = validation_dataset.select(range(num_samples))
    metrics, fine_tuned_model, fine_tuned_tokenizer = tuning(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        hyperparameters_dict["1__num_epochs"],
        tokenizer,
        validation_dataset,
    )
    output_dir = task.context().output

    fine_tuned_model.save_pretrained(output_dir / "fine_tuned_model/")
    fine_tuned_tokenizer.save_pretrained(output_dir / "fine_tuned_tokenizer/")

    metrics["epoch_num"].append(hyperparameters_dict["1__num_epochs"])
    metrics["learning_rate"].append(hyperparameters_dict["1__learning_rate"])
    metrics["batch_size"].append(hyperparameters_dict["0__batch_size"])

    end_time = time.time()
    metrics["cost"] = end_time - start_time

    with open(output_dir / "metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file)

    print("fine tuning completed")

    return output_dir


@task(cache=True, cache_key_file="hparams")
def model_distillation(
    fine_tuned_model_path: DirectoryPath,
    data_path: DirectoryPath,
    dataset: DirectoryPath,
    *,
    hparams: FilePath,
) -> DirectoryPath:
    """Distillation Stage."""
    start_time = time.time()
    model_name = "t5-small"
    hyperparameters_dict = load_hyperparameters(dataset)

    # Load your fine-tuned t5-small teacher model and tokenizer
    teacher_model = T5ForConditionalGeneration.from_pretrained(
        fine_tuned_model_path / "fine_tuned_model/"
    ).to("cuda")

    # Load T5-tiny student model
    tokenizer = T5Tokenizer.from_pretrained(
        model_name, d_model=128, d_ff=512, d_kv=64, num_layers=2
    )
    student_config = T5Config.from_pretrained(
        model_name, d_model=128, d_ff=512, d_kv=64, num_layers=2
    )
    student_model = T5ForConditionalGeneration(student_config).to("cuda")

    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=hyperparameters_dict["2__learning_rate"]
    )

    # Define your training & validation dataset and dataloader
    train_dataloader = torch.load(data_path / "training_data")
    val_dataloader = torch.load(data_path / "validation_data")
    validation_dataset = datasets.load_from_disk(dataset / "validation_data")
    output_dir = task.context().output

    if num_samples > 0 and num_samples < 13368:
        validation_dataset = validation_dataset.select(range(num_samples))

    metrics, distilled_model, distilled_tokenizer = distillation(
        teacher_model,
        student_model,
        train_dataloader,
        val_dataloader,
        optimizer,
        hyperparameters_dict["2__num_epochs"],
        tokenizer,
        validation_dataset,
        hyperparameters_dict["2__temperature"],
        output_dir,
    )

    distilled_model.save_pretrained(output_dir / "distilled_model")
    distilled_tokenizer.save_pretrained(output_dir / "distilled_tokenizer")

    metrics["epoch_num"].append(hyperparameters_dict["2__num_epochs"])
    metrics["learning_rate"].append(hyperparameters_dict["2__learning_rate"])
    metrics["batch_size"].append(hyperparameters_dict["0__batch_size"])
    end_time = time.time()
    metrics["cost"] = end_time - start_time
    with open(output_dir / "metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file)

    print("Distillation has been completed")

    return output_dir


@task(cache=True, cache_key_file="hparams")
def model_inference(
    dataset: DirectoryPath,
    data_path: DirectoryPath,
    fine_tuned_model_path: DirectoryPath,
    distilled_model_path: DirectoryPath,
    *,
    hparams: FilePath,
) -> DirectoryPath:
    start_time = time.time()
    fine_tuned_model = T5ForConditionalGeneration.from_pretrained(
        fine_tuned_model_path / "fine_tuned_model/"
    )
    fine_tuned_tokenizer = T5Tokenizer.from_pretrained(
        fine_tuned_model_path / "fine_tuned_tokenizer/"
    )
    # Load the distilled model
    distilled_model = T5ForConditionalGeneration.from_pretrained(
        distilled_model_path / "distilled_model/"
    )
    # Quantize the distilled model
    quantized_distilled_model = torch.quantization.quantize_dynamic(
        distilled_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_distilled_tokenizer = T5Tokenizer.from_pretrained(
        distilled_model_path / "distilled_tokenizer/"
    )
    test_dataset = datasets.load_from_disk(data_path / "testing_data")
    test_data = test_dataset["article"]
    reference_summaries = test_dataset["highlights"]

    # Inference for fine-tuned model
    fine_tuned_metrics = inference(
        fine_tuned_model, fine_tuned_tokenizer, test_data, reference_summaries
    )

    # Inference for quantized model
    quantized_metrics = inference(
        quantized_distilled_model,
        quantized_distilled_tokenizer,
        test_data,
        reference_summaries,
    )

    output_dir = task.context().output

    metrics = {
        "fine_tuned_metrics": fine_tuned_metrics,
        "quantized_metrics": quantized_metrics,
        "cost": 0,
    }
    end_time = time.time()
    metrics["cost"] = end_time - start_time

    with open(output_dir / "metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file)

    print("Inference has been completed")

    return dataset


@task(cache=True, cache_key_file="hparams")
def generate_output(
    dataset: DirectoryPath,
    data_path: DirectoryPath,
    fine_tuning_output: DirectoryPath,
    distillation_output: DirectoryPath,
    inference_output: DirectoryPath,
    *,
    hparams: FilePath,
) -> DirectoryPath:
    """Output Generation."""
    # with open(dataset / "initial_hparams.json") as initial_hparams:
    #     initial_hps = json.load(initial_hparams)

    final_metrics = {"cost": "", "obj": 0}

    # with open(dataset / "hparams.json") as _hparams:
    #     hyperparmeters = json.load(_hparams)

    # hp_dataset = hyperparmeters["dataset"]

    with open(data_path / "metrics.json") as _metrics:
        data_metrics = json.load(_metrics)

    with open(fine_tuning_output / "metrics.json") as _metrics:
        fine_tuning_metrics = json.load(_metrics)

    with open(distillation_output / "metrics.json") as _metrics:
        distillation_metrics = json.load(_metrics)

    with open(inference_output / "metrics.json") as _metrics:
        inference_metrics = json.load(_metrics)

    final_metrics["cost"] = [
        data_metrics["cost"],
        fine_tuning_metrics["cost"],
        distillation_metrics["cost"],
        # inference_metrics["cost"],
    ]

    final_metrics["obj"] = distillation_metrics["rouge_scores"][0]["rougeLsum"]

    # obj_validation_loss = metrics["validation_loss"]
    # obj_validation_loss_avg = sum(obj_validation_loss) / len(obj_validation_loss)

    # hp_dataset["obj"].append(obj_validation_loss_avg)

    # initial_hps["dataset"] = hp_dataset

    output_dir = task.context().output

    with (output_dir / "hparams_file.json").open("w") as f:
        json.dump(final_metrics, f)

    return output_dir


@pipeline
def t5_fine_tuning(
    dataset: DirectoryPath,
    hparams: FilePath,
    data_preproc_hp: FilePath,
    tuning_hp: FilePath,
    distillation_hp: FilePath,
) -> DirectoryPath:
    """Main Pipeline."""
    data_preprocessing_output = data_preprocessing(dataset, data_preproc_hp)
    fine_tuning_output = fine_tuning(data_preprocessing_output, dataset, tuning_hp)
    distillation_output = model_distillation(
        fine_tuning_output, data_preprocessing_output, dataset, distillation_hp
    )
    inference_output = model_inference(
        dataset,
        data_preprocessing_output,
        fine_tuning_output,
        distillation_output,
        hparams,
    )
    output = generate_output(
        dataset,
        data_preprocessing_output,
        fine_tuning_output,
        distillation_output,
        inference_output,
        hparams,
    )

    return output
