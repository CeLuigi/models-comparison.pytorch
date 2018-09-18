Available scripts:
* **compute_computational_complexity.py:** computes the GFlops per model and stores the results into a JSON file;
* **compute_memory_usage.py:** estimates the GPU-memory usage per model by varying the batch size (requires the installation of [GPUtil module](https://github.com/anderskm/gputil)). It stores the results into a JSON file;
* **compute_inference_time.py:** computes inference time per model by varying the batch size (resulting inference time is the average of 10 runs);
* **compute_accuracy_rate.py:** estimates performances on Imagenet validation set per model and stores the results into a JSON file.
