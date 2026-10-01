Spark-Based Benchmarking of Computer Vision Architectures
Source code and raw experimental logs for:

Performance Evaluation of Advanced Computer Vision Models in Big Data Processing EnvironmentsHabib Matin, Keyhan Khamforoosh, Mehdi Taghipanah16th International Conference on Computer and Knowledge Engineering (ICCKE 2026), Ferdowsi University of Mashhad — Paper 1073

What This Is
An empirical benchmark of nine CNN and Vision Transformer architectures for imageclassification, evaluated under an Apache Spark-managed pipeline on a singleCPU-only workstation (Spark local mode, two logical workers). All measurements —accuracy, throughput, and latency — are reported with their run-to-run statistics,and every number in the paper can be traced back to the raw logs in results/.

Scope statement (important): the pipeline is a single-machine pseudo-cluster:Spark partitions the per-architecture evaluation tasks across two logical workers,while training and inference run on the host CPU with PyTorch. The reported figurescharacterize this simulated distributed pipeline — not multi-node scaling.

Results at a Glance
Role	Model	Accuracy (%)	Throughput (img/s)
Best accuracy	ConvNeXt-Tiny	81.64	12.96
Runner-up	ViT-Base-16	80.56 (n=1)	5.57
Best balance	EfficientNet-B0	61.95	26.18
Fastest	MobileNet-V3 (Small)	47.86	85.40
Accuracy = mean over two runs on Food-101 + ImageNet-Mini; throughput = mean overthree dataset workloads. Full tables: see the paper.

Repository Structure
Path	Purpose
spark_benchmark.py	Main pipeline — interactive selection of datasets × models; Spark parallelizes the per-architecture evaluation tasks (2 workers); per-architecture training is linear probing on a frozen ImageNet-pretrained backbone
repeatability_benchmark.py	Dedicated timing study: warm-up + three timed trials, forward-pass-only latency (ResNet50, EfficientNet-B0, MobileViT-S)
tools/rebuild_flowers_by_class.py	Restructures the flat 102 Flowers archive into 102 class folders using imagelabels.mat (see Flowers-102 note below)
results/raw_benchmark_runs.csv	Raw per-run logs behind Tables VI and VIII (two full runs per model per dataset; ViT-Base-16 has one)
results/repeatability_results.csv	Raw trials behind Table IX
Environment
Component	Specification
CPU	Intel Xeon Gold 6430, 4 logical cores (CPU-only — no GPU in any experiment)
Memory	16 GB
OS	Microsoft Windows Server 2022 (64-bit)
Python	3.10.9
DL framework	PyTorch 2.5.1 + torchvision 0.20.1 (CPU execution)
Distributed engine	Apache Spark 4.2.0, local mode, 2 logical workers (Java 21)
Install dependencies with:

pip install -r requirements.txt
(Java 21 is required on the PATH for Spark.)

Datasets
Place datasets under DataSet/ with the following layout:

DataSet/├── 102flowers_by_class/          # 102 class folders — build with tools/rebuild_flowers_by_class.py├── food-101/food-101/images/     # 101 class folders└── imagenet-mini/train/          # 1000 wnid folders
102 Flowers (Oxford-102): download 102flowers.tgz, imagelabels.mat, setid.matfrom the official page, then runtools/rebuild_flowers_by_class.py.
Food-101: from the official sourceor Kaggle mirror.
ImageNet-Mini: the publicly circulated Kaggle subset (1,000 classes, 34,745 imagesin its train partition). Caveat: this subset may overlap the ImageNet-1k pretrainingdata of the evaluated backbones; accuracies on it are therefore read as upper boundsin the paper.
Flowers-102 note. The public archive stores all 8,189 images in a single flatdirectory. Deriving class labels from the directory structure without restructuringyields a single class and silently produces meaningless 100% accuracy. This repositorytherefore (a) provides tools/rebuild_flowers_by_class.py, and (b) the main loader nowasserts the expected class count:

num_classes = len(full_dataset.classes)assert num_classes >= 5, (    f"[FATAL] {d_name}: only {num_classes} class folder(s) found — "    "dataset structure is broken; aborting.")
In the paper, Flowers-102 is used solely as a uniform operational workload forthroughput/latency profiling; classification accuracy is reported only on Food-101and ImageNet-Mini.

Usage
python spark_benchmark.py
The script presents two interactive menus (dataset(s) and model(s)); Spark then runsthe selected evaluation tasks in parallel across two workers. Results are appended topaper_results/Vision_Benchmark_Results.csv, with per-run rows containing accuracy,precision/recall/F1, throughput, pipeline latency, parameter counts, peak RAM, andper-epoch training accuracy.

Training protocol (identical for all models, Table IV of the paper):

Parameter	Value
Initial weights	ImageNet-pretrained, backbone frozen
Trained part	final classification head only (linear probing)
Optimizer	Adam, lr = 1e-3 (fixed)
Epochs	3
Batch size	64
Split	80/20 random at the unique-image level, seed 42
Augmentation	none (resize to 224×224 + ImageNet normalization)
Methodology & Statistics
Runs: two full runs per model per dataset; results in the paper are mean ± std.ViT-Base-16 has a single available run (marked n = 1 in the paper).
Throughput/latency (pipeline): measured over the whole test partition inside theSpark worker task; latency is reported as 1000/throughput and therefore includesdata-loading overhead.
Repeatability study: separate forward-pass-only benchmark with explicit warm-upand three timed trials; standard deviations stayed below 1.1% of the mean.
Efficiency Score: (mean top-1 accuracy × mean throughput)/1000; the paper reportsa sensitivity analysis over alternative accuracy/throughput weightings.
Paper-Table ↔ Artifact Map
Paper table	Source
Table VI (accuracy, mean ± std)	results/raw_benchmark_runs.csv
Table VII (published comparison)	reference results from the cited papers
Table VIII (throughput/latency)	results/raw_benchmark_runs.csv
Table IX (repeatability)	results/repeatability_results.csv
Tables X–XII (derived)	computed from the above
Provenance Notes
MobileNet-V3 denotes the Small variant (2.5M parameters).
The repeatability study additionally includes MobileViT-S loaded via timm, as alightweight hybrid outside the main nine models; its accuracy columns in the raw CSVreflect untrained heads and are not used (only the timing columns are reported).
Energy consumption is not measured or reported: system-level energy monitoring islisted as future work in the paper.
Citation
@inproceedings{matin2026performance,  title     = {Performance Evaluation of Advanced Computer Vision Models in Big Data Processing Environments},  author    = {Matin, Habib and Khamforoosh, Keyhan and Taghipanah, Mehdi},  booktitle = {16th International Conference on Computer and Knowledge Engineering (ICCKE)},  year      = {2026},  address   = {Mashhad, Iran}}
License
MIT — see LICENSE.

Contact
Habib Matin — Department of Computer Engineering, Sanandaj Branch, Islamic Azad University, Iran.Issues and questions are welcome via the issue tracker.

requirements.txt
text

torch==2.5.1
torchvision==0.20.1
pyspark==4.2.0
pandas
numpy
scikit-learn
psutil
scipy
timm
