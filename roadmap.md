# Claro.AI – Roadmap

🧠 Project Start: March 2026  
🎯 Goal: Build an MVP that allows users to visually build Hugging Face model apps and run/export them via Colab.

---

## 🪜 Phase 0: Prep Phase (Before March)

- [x] Create GitHub repo (`claro-ai`)
- [x] Design logo and branding
- [x] Mock up UI (drag zone, HF panel, code cells, etc.)
- [ ] Set up project folder (`designs/`, `docs/`, `ui/`)
- [ ] Plan tech stack (React + Tailwind, Monaco Editor, Hugging Face API, etc.)

---

## 🚀 Phase 1: Core MVP (March – April)

> Goal: A working prototype where you can drag a Hugging Face model in, build basic input/output flow, and export runnable code.

### Week 1–2: App Setup & UI Skeleton

- [ ] Scaffold React app (`vite` or `next.js`)
- [ ] Create main layout:
  - Sidebar (file tree or model list)
  - Main area (drag-and-drop blocks)
  - Right panel (model search or block settings)
- [ ] Add TailwindCSS or Chakra UI
- [ ] Implement dark/light theme toggle

---

### Week 3: Hugging Face Model Search Panel

- [ ] Integrate Hugging Face Hub API (use `@huggingface/hub`)
- [ ] Search models by name/task (text-gen, image-classify, etc.)
- [ ] Display model name, ID, tags, and thumbnail
- [ ] Drag or click to add model block to canvas

---

### Week 4–5: Block/Cell System

- [ ] Create draggable blocks:
  - Model
  - Input (text, image, number)
  - Output (text, label, image)
- [ ] Allow blocks to be added, removed, and reordered
- [ ] Optional: Snap to grid / visual lines between blocks
- [ ] Add ability to edit block parameters (like temperature)

---

### Week 6: Code Generation Engine

- [ ] Set up internal data structure (JSON?) to represent block flow
- [ ] Write Gradio + Python templates using dynamic strings
- [ ] Convert current layout into runnable code (code preview pane)

---

## 📦 Phase 2: Output & Export Features (May)

> Goal: Turn Claro.AI into something shareable and runnable on the cloud.

- [ ] Add Monaco Editor or syntax-highlighted preview
- [ ] Generate `.ipynb` file from code output
- [ ] Add “Open in Colab” button with pre-filled notebook
- [ ] Optional: Export project as JSON

---

## ✨ Phase 3: Polish & Test (June)

> Goal: Make it clean, stable, and fun to use.

- [ ] Add example templates (e.g. “Chatbot Starter”, “Image Classifier”)
- [ ] Handle invalid model/task connections
- [ ] Improve error messages / form validation
- [ ] Optimize UI responsiveness
- [ ] Theme polish (spacing, transitions, icons)

---

## 🌍 Phase 4: Community & Launch (Summer 2026)

> Goal: Get the first users. See what breaks. Iterate fast.

- [ ] Create landing page (Vercel)
- [ ] Publish MVP demo video
- [ ] Launch on X, Reddit, Hacker News, Product Hunt
- [ ] Add GitHub discussions / feedback form
- [ ] Get user feedback → plan v2

---

## 💡 Future Ideas

- [ ] Drag multiple models and chain them (e.g. vision → text)
- [ ] Deploy to Hugging Face Spaces directly
- [ ] Offline mode (desktop Electron build?)
- [ ] Custom block editor
- [ ] AI suggestions for block sequences

---

## 🛠 Tech Stack (Planned)

| Area | Tech |
|------|------|
| Frontend | React, TailwindCSS, Vite |
| Drag UI | `dnd-kit`, `react-flow`, or custom |
| Model Search | Hugging Face Hub API |
| Code Preview | Monaco Editor / syntax-highlighted view |
| Export | `.ipynb` notebook or `.py` + Colab link |
| Hosting | Vercel / Netlify |
| Design | Figma / Illustrator / handwritten ideas |

---

## 🧑‍🎓 Target Users

- Students learning AI
- Teachers making interactive demos
- Curious devs trying out models
- Beginner researchers who don’t want to write boilerplate

---

Claro.AI – Drag. Drop. Done.  
Stay claro 😎
