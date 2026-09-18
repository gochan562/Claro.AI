<div align="center">

# Claro.AI


A browser-based AI workbench for discovering Hugging Face models, building notebooks, running inference, and fine-tuning models without wiring together every piece of ML infrastructure by hand.

</div>

### 🚀 What is Claro.AI?

Claro.AI turns AI experimentation into a visual notebook workflow.

Instead of jumping between model cards, Python scripts, training configs, GPU services, and inference code, Claro.AI brings the main pieces into one browser workspace.

Discover → Configure → Run → Train → Experiment

The project is designed around a simple idea:

Make the path from an AI idea to a working experiment shorter and easier to understand.

### ✨ Features

**📓 Visual AI Notebooks**

 Build notebooks directly in the browser using different cell types:

- Code Cells : Python code and experiments

- Markdown Cells : notes and documentation

- Model Cells : discover and configure Hugging Face models

- Parameter Cells : configure inference parameters visually

- Training Cells : configure and monitor fine-tuning jobs

Notebook workspaces are persisted locally in the browser.

### 🤗 Hugging Face Integration

**Claro.AI connects directly to the Hugging Face ecosystem for model and dataset discovery.**

Model search includes task-oriented information and metadata such as:

- pipeline / task

- downloads

- likes

- tags

- update information

Models and datasets can then be incorporated into notebook workflows.

### 🧠 Architecture-Aware Inference

**Claro.AI does more than pass a model name to a generic inference endpoint.**

Its model-loading pipeline can inspect model architecture, determine a loading strategy, and generate the appropriate loader / inference workflow.

This allows different Hugging Face model families to be handled through the same visual notebook interface.

### 🏋️ Fine-Tuning

**The Training Cell provides a visual interface for configuring model training.**

Current training options include:

- model and dataset selection

- epochs

- batch size

- learning rate

- validation split

- sample limits

- training-step limits

- training-time limits

- full fine-tuning

- LoRA fine-tuning

Supported task types currently include:

- text-generation
- text-classification
- image-classification
- token-classification

Training jobs expose live progress and metrics such as loss, epoch, step, learning rate, elapsed time, and GPU status.

⚡ Local + Remote Execution

Claro.AI separates the notebook experience from the compute infrastructure.

Simple Python can run locally in the browser through Pyodide, while ML workloads can be routed to remote GPU infrastructure.

Current remote inference backends include:

Hugging Face ZeroGPU

Modal

The backend uses a provider abstraction so the notebook workflow does not need to know which GPU service is executing the job.

🏗️ Architecture

Claro.AI is split into several cooperating layers:

<p align="center">
  <img src="docs/architecture-overview.png" alt="Claro.AI architecture overview" width="900">
</p>

Detailed Architecture ↓

The detailed diagram shows the current relationships between the browser workbench, gateway/composition layer, training and artifact system, Python runtimes, Hugging Face, and remote GPU providers.

<p align="center">
  <img src="docs/architecture-detailed.png" alt="Claro.AI detailed architecture" width="900">
</p>


### 🧩 Core Components

**Component**

Role

index.html -------- Landing interface

dashboard.html ---- Main browser workbench

dashboard-app.js -- Notebook and dashboard application logic

training_ui.js ---- Training controls, presets, and validation

server.js --------- Express API gateway

gpu_backends.js --- GPU provider abstraction

training_backend.js - Training job orchestration

training_runner.py - Training / fine-tuning execution

inference_runner.py - Python inference runtime

trainer.py --------- Training utilities

pyodide_local.js --- Browser-side Python bridge

hf_loader/ --------- Hugging Face model-loading logic

space/ ------------- Remote Hugging Face Space components

### 🛠️ Tech Stack

**Frontend**

HTML, CSS, Vanilla JavaScript, CodeMirror, Pyodide, Marked, DOMPurify, Chart.js, xterm.js

**Backend**

Node.js, Express

**ML / Python**

Python, PyTorch, Hugging Face Transformers, Hugging Face Datasets, Accelerate, PEFT

**Compute**

Hugging Face ZeroGPU

Modal (currently not available)

Local browser Python through Pyodide

### 🚀 Getting Started

Requirements

- Node.js

- Python 3

- Python packages listed in requirements.txt

1. Clone

```
git clone https://github.com/gochan562/Claro.AI.git
cd Claro.AI
```

2. Install dependencies
```
npm install
python3 -m pip install -r requirements.txt
```
3. Configure environment variables
```
cp .env.example .env
```
The repository provides environment variables for the server, Python runtime, Hugging Face, ZeroGPU, Modal, and training configuration.

For example:
```
PORT=5000
FRONTEND_ORIGIN=http://localhost:5000

TRAINING_PROVIDER=local
PYTHON_BIN=python3

GPU_PROVIDER=zerogpu
ZEROGPU_SPACE=Gochan562/claro_ai_gpu
```
4. Start Claro.AI
```
npm start
```
Then open:
```
http://localhost:5000
```
### 🔐 Security

Claro.AI is an experimental developer tool.

Some execution paths can process user-supplied code or route workloads to remote execution infrastructure. A public deployment should therefore be treated as infrastructure that needs additional isolation and access control.

For an internet-facing deployment, add appropriate:

- authentication / authorization

- execution sandboxing

- resource limits

- network isolation

- artifact controls

- rate limiting

Do **not** expose unrestricted code execution to untrusted users without additional security boundaries.

### 🧪 Current Status

Claro.AI is an evolving prototype.

The core workflow is implemented across model discovery, notebook composition, inference, training orchestration, and remote GPU execution, while some product areas remain under development.

Current development is focused on improving model compatibility, training workflows, infrastructure abstraction, and the overall notebook experience.

### 🗺️ Roadmap

Potential future work includes:

- [ ] broader Hugging Face task support

- [ ] improved model compatibility detection

- [ ] more training presets

- [ ] stronger artifact management

- [ ] richer experiment tracking

- [ ] dataset upload workflows

- [ ] persistent cloud workspaces

- [ ] improved multi-user isolation

- [ ] more visual workflow components

- [ ] simpler deployment

### 💡 Why Claro.AI?

Traditional AI experimentation can involve a lot of glue code:

Find a model
    ↓
Understand its architecture
    ↓
Figure out the loader
    ↓
Install dependencies
    ↓
Write inference code
    ↓
Find compute
    ↓
Configure training
    ↓
Run and monitor experiments

Claro.AI tries to compress that into a single workspace.

The goal is not to hide the underlying ML systems.

It is to make them easier to work with.

### 🔗 Links

Repository: https://github.com/gochan562/Claro.AI

Hugging Face: https://huggingface.co/

<div align="center">

Claro.AI

Drag, drop, run.

</div>
