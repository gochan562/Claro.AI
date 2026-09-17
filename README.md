# Claro.AI

### Drag, drop, run.

Claro.AI is a browser-based AI workbench that makes working with Hugging Face models feel more like using a visual notebook than wiring together ML code by hand.

It brings **model discovery, notebooks, inference, parameters, and fine-tuning** into one interface — designed especially for students, beginners, and developers who want to experiment with AI without spending most of their time on boilerplate.

> **Goal:** make AI development easier to understand, experiment with, and build.

---

## ✨ What Claro.AI Does

Claro.AI combines a visual notebook with Hugging Face and remote GPU backends.

### 📓 Visual Notebooks

Create and manage multiple notebooks directly in the browser.

Each notebook can contain:

* 💻 **Code Cells** — write and execute Python
* 📝 **Markdown Cells** — notes, explanations, and documentation
* 🤖 **Model Cells** — select and load Hugging Face models
* 🎛️ **Parameter Cells** — configure model parameters visually
* 🏋️ **Training Cells** — fine-tune models on datasets

Notebooks are persisted in the browser using `localStorage`, so your workspace can survive page reloads without requiring a database.

---

## 🤗 Hugging Face Integration

Claro.AI connects directly to the Hugging Face ecosystem.

### Model discovery

Search and browse models from the Hugging Face Hub, with filters for tasks such as:

* Text generation
* Text classification
* Token classification
* Image classification
* Question answering
* Summarization
* Translation
* Computer vision
* Speech
* And more

Model cards display useful metadata such as downloads, likes, tags, pipeline type, and update date.

Models can be selected directly into a Model Cell or dragged into the notebook workflow.

### Dataset discovery

The dashboard also provides Hugging Face dataset search with sorting and filtering options.

---

## 🧠 Inference

Model Cells can generate architecture-aware loader and inference code based on the selected Hugging Face model.

Claro.AI supports task-specific inference workflows rather than treating every model as a generic text generator.

The application can route inference through a configurable GPU backend.

### GPU backends

**Hugging Face ZeroGPU** is the default backend.

**Modal** is also supported as an alternative backend.

```text
Browser
   │
   ▼
Claro.AI Dashboard
   │
   ├── Local Python → Pyodide
   │
   └── Remote inference
          │
          ├── Hugging Face ZeroGPU
          └── Modal
```

The GPU provider can be selected through environment variables:

```env
GPU_PROVIDER=zerogpu
```

or:

```env
GPU_PROVIDER=modal
```

---

## 🏋️ Model Fine-Tuning

Claro.AI includes a real training pipeline built around the Hugging Face training ecosystem.

The Training Cell supports:

* Dataset selection
* Model selection
* Epochs
* Batch size
* Learning rate
* Validation split
* Maximum samples
* Maximum training steps
* Training time limits
* Full fine-tuning
* LoRA fine-tuning
* Automatic training-method selection

Currently supported training task types include:

* `text-generation`
* `text-classification`
* `image-classification`
* `token-classification`

### Training monitoring

Training jobs expose live information such as:

* Current epoch
* Current step
* Training loss
* Evaluation loss
* Learning rate
* Elapsed time
* Progress
* GPU status

Training metrics are taken from the actual Hugging Face training process rather than simulated loss curves.

After training, Claro.AI can keep the resulting artifacts and create a testing workflow for the trained model.

---

## 🧪 Built-in Training Presets

The Training Cell includes beginner-friendly presets for models and datasets.

Examples include:

| Model       | Task                 | Dataset        |
| ----------- | -------------------- | -------------- |
| DistilBERT  | Text classification  | IMDb / AG News |
| MobileNetV2 | Image classification | Beans          |
| ResNet-18   | Image classification | Beans          |

The training UI also validates model/dataset/task compatibility before starting a job.

---

## 🖥️ Browser-Side Python

Python code can run locally in the browser through **Pyodide**.

This means simple Python experiments do not always need a remote GPU.

The notebook interface also uses:

* **CodeMirror** for code editing
* **Marked** for Markdown rendering
* **DOMPurify** for sanitizing rendered Markdown
* **Chart.js** for training metrics
* **xterm.js** for the GPU console

---

## 🧩 Architecture

Claro.AI intentionally keeps the frontend lightweight.

### Frontend

* HTML
* CSS
* Vanilla JavaScript
* CodeMirror
* Pyodide

There is currently no React/Vue build pipeline or frontend bundler.

### Backend

* Node.js
* Express
* Python
* Hugging Face Transformers
* Hugging Face Datasets
* Accelerate
* PEFT
* PyTorch

### GPU abstraction

The backend separates GPU infrastructure from the notebook UI:

```text
                    Claro.AI
                       │
                Express Server
                       │
          ┌────────────┴────────────┐
          │                         │
      Inference                  Training
          │                         │
   gpu_backends.js         training_backend.js
          │                         │
     ┌────┴────┐             ┌──────┴──────┐
     │         │             │             │
  ZeroGPU   Modal        Local Python    ZeroGPU
```

---

## 📁 Project Structure

```text
Claro.AI/
├── public/
│   ├── index.html
│   ├── dashboard.html
│   ├── dashboard-app.js
│   ├── training_ui.js
│   └── ...
│
├── server.js
├── gpu_backends.js
├── training_backend.js
├── training_runner.py
├── inference_runner.py
├── trainer.py
├── requirements.txt
├── package.json
├── .env.example
└── README.md
```

### Important files

| File                  | Purpose                         |
| --------------------- | ------------------------------- |
| `server.js`           | Express server and API routes   |
| `gpu_backends.js`     | GPU provider abstraction        |
| `training_backend.js` | Training job management         |
| `training_runner.py`  | Actual model training           |
| `inference_runner.py` | Inference for trained models    |
| `training_ui.js`      | Training presets and validation |
| `dashboard-app.js`    | Main notebook/application logic |
| `requirements.txt`    | Python ML dependencies          |
| `.env.example`        | Environment configuration       |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/gochan562/Claro.AI.git
cd Claro.AI
```

### 2. Install Node.js dependencies

```bash
npm install
```

### 3. Install Python dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

The default configuration uses Hugging Face ZeroGPU:

```env
PORT=5000
FRONTEND_ORIGIN=http://localhost:5000

TRAINING_PROVIDER=local
PYTHON_BIN=python3

GPU_PROVIDER=zerogpu
ZEROGPU_SPACE=Gochan562/claro_ai_gpu
```

For a private Hugging Face Space, a server-side token can also be configured:

```env
ZEROGPU_API_TOKEN=your_token_here
```

Never commit `.env` or other secrets to Git.

### 5. Start the server

```bash
npm start
```

Then open:

```text
http://localhost:5000
```

---

## ⚙️ Environment Configuration

Common variables include:

```env
# Server
PORT=5000
FRONTEND_ORIGIN=http://localhost:5000

# Training
TRAINING_PROVIDER=local
PYTHON_BIN=python3

# GPU
GPU_PROVIDER=zerogpu
ZEROGPU_SPACE=Gochan562/claro_ai_gpu

# Optional ZeroGPU training configuration
ZEROGPU_TRAIN_API=
ZEROGPU_TRAIN_TIMEOUT_MS=
ZEROGPU_API_TOKEN=
ZEROGPU_TRAIN_MAX_ARTIFACT_BYTES=

# Optional Modal backend
MODAL_RUN_URL=
MODAL_STREAM_URL=
MODAL_AUTH_SECRET=

# Optional Hugging Face authentication
HF_TOKEN=
```

See `.env.example` for the complete configuration.

---

## 🔐 Security Note

Claro.AI is an experimental developer tool and should be deployed carefully.

Some server endpoints can forward user-supplied code to a configured remote execution backend. A publicly exposed deployment therefore should **not** be treated as a secure multi-user sandbox by default.

For an internet-facing deployment, add appropriate authentication, authorization, isolation, resource limits, and infrastructure-level sandboxing before allowing untrusted users to execute arbitrary code.

---

## 🧭 Current Status

Claro.AI is an **active prototype / experimental project**.

The main workflow is already implemented:

```text
Discover model
      ↓
Add model to notebook
      ↓
Configure parameters
      ↓
Run inference
      ↓
Choose dataset
      ↓
Fine-tune
      ↓
Monitor training
      ↓
Test trained model
```

The project is still evolving, and some dashboard areas such as **Payment Plans** and **Settings** are currently placeholders.

---

## 🗺️ Roadmap

Potential areas for future development:

* Better model loading and compatibility detection
* More training presets
* More Hugging Face task types
* Improved GPU scheduling
* Better artifact management
* Dataset upload workflows
* Richer experiment tracking
* Persistent cloud workspaces
* More robust multi-user isolation
* More visual workflow components
* Simpler deployment for students

---

## 💡 Why Claro.AI?

AI development often starts with a surprisingly large amount of setup:

```text
Find a model
    ↓
Understand its architecture
    ↓
Install dependencies
    ↓
Write loading code
    ↓
Configure inference
    ↓
Find a GPU
    ↓
Write training code
    ↓
Monitor the run
```

Claro.AI is an attempt to compress that process into a visual workflow:

```text
Pick → Configure → Run → Experiment
```

The idea is not to hide the underlying technology.

It is to make the technology easier to reach.

---

## 📜 License

This project is currently distributed under the license specified in the repository.

See the repository for the latest licensing information.

---

## 🔗 Relevant links

* **GitHub:** https://github.com/gochan562/Claro.AI
* **Hugging Face:** https://huggingface.co/
* **Project:** Claro.AI — Drag & Drop AI GUI
