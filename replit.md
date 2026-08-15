# Claro.AI - Drag & Drop AI GUI

## Overview

Claro.AI is a web-based AI GUI application featuring a drag-and-drop interface. The project consists of a multi-page web application with a landing page, authentication system (login/signup), and a dashboard interface. The application uses a modern dark-themed UI with animated visual elements and glassmorphism design patterns.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Stack**: Pure HTML/CSS/JavaScript
- No framework dependencies detected
- Vanilla JavaScript for client-side interactions
- Static HTML pages for different application views

**Design Pattern**: Multi-page application (MPA)
- Separate HTML files for each major view (landing, login, signup, dashboard)
- Shared styling considerations with dark theme
- Centralized JavaScript (script.js) and CSS (style.css) files currently empty, suggesting future consolidation

**UI/UX Approach**: 
- **Dark Mode First**: Deep blue/black gradient backgrounds (#050813, #0a0f2e)
- **Glassmorphism**: Backdrop blur effects with semi-transparent backgrounds
- **Animation**: CSS keyframe animations for floating blob elements
- **Responsive Design**: Viewport meta tags for mobile compatibility

**Rationale**: The pure HTML/CSS/JS approach provides maximum flexibility for prototyping and eliminates build tool complexity. However, this creates potential code duplication across pages. A future migration to a framework like React or Vue could improve maintainability.

### Page Structure

**Landing Page (index.html)**:
- Hero section with animated canvas background
- Sticky header with navigation
- Logo branding with text shadow effects

**Authentication Pages (Login.html, signup.html)**:
- Centered form layout
- Animated background blobs for visual interest
- Consistent styling with radial gradients

**Dashboard (dashboard.html)**:
- Sidebar navigation (240px fixed width)
- User profile section in sidebar with dynamic username and avatar
- Flexbox-based layout with horizontal content area
- Dark themed (#0f172a sidebar, #0a101f main)
- **Authentication Integration** (November 2025):
  - Firebase onAuthStateChanged for user session detection
  - Automatic redirect to login page if not authenticated
  - Email/password users: Display username from signup, geometric avatar from DiceBear API
  - OAuth users (Google/GitHub): Display provider username and profile photo
  - Logout functionality integrated in sidebar
- **Notebook Interface** (November 2025):
  - Multi-notebook management system with list and edit views
  - Create, rename, delete, and switch between multiple named notebooks
  - Double-click navigation from list view to edit mode
  - Three cell types: Code (Python), Markdown, and Model (Hugging Face)
  - **Python Execution**: Pyodide (in-browser Python runtime) for executing Python code
  - **Output Capture**: Captures stdout/stderr from Python execution
  - Cell execution: Run individual cells, all cells, or markdown-only cells
  - Hugging Face integration: Browse, search, and insert models from HF Hub
  - Drag-and-drop: Drag models from picker into model cells
  - **localStorage Registry**: Multi-notebook persistence with backward compatibility
  - **Legacy Migration**: Automatically migrates old single-notebook format to new registry
  - Security: DOMPurify sanitization for markdown, sandboxed Python execution

**Pros**: 
- Simple deployment (static files)
- No build process required
- Easy to understand structure

**Cons**:
- Code duplication across pages
- No shared component system
- Manual state management required

### Asset Management

**Current State**: Empty centralized files (script.js, style.css)
- Inline styles currently used in all HTML files
- No external CSS frameworks detected
- Placeholder for future JavaScript functionality

**Future Consideration**: Consolidating styles into style.css and shared logic into script.js would reduce duplication and improve maintainability.

## External Dependencies

### Currently Integrated

**Firebase Authentication** (November 2025):
- Firebase SDK v10.12.1 integrated for user authentication
- Supports email/password authentication with username
- Google OAuth integration
- GitHub OAuth integration
- Used in Login.html, signup.html, and dashboard.html

**DiceBear Avatar API**:
- Geometric avatar generation for email/password users
- Uses user email as seed for consistent avatar generation

**Hugging Face Hub API**:
- Integration with HF Hub API for browsing and searching models
- Real-time model list fetching with task-based filtering
- Support for popular tasks: text-generation, classification, summarization, etc.
- Displays model metadata: downloads, likes, tags, pipeline type

**Marked.js** (v9.x):
- Markdown parsing and rendering for markdown cells
- Coupled with DOMPurify for XSS protection

**DOMPurify** (v3.0.6):
- HTML sanitization for markdown cell outputs
- Prevents stored XSS attacks in persisted notebooks

**Pyodide** (v0.24.1):
- In-browser Python runtime for executing Python code cells
- Approximately 10MB bundle size, loaded asynchronously on dashboard initialization
- Supports standard Python libraries and packages
- Enables Python execution without requiring a backend server
- Integrated with stdout/stderr capture for displaying execution results

### Potential Future Dependencies

Based on the application's purpose (AI GUI with drag-and-drop), likely future integrations include:

1. **AI/ML Services**: OpenAI API, Anthropic Claude, or similar for AI functionality
2. **Backend API**: Node.js/Express, Python/Flask, or similar for AI orchestration and data persistence
3. **Database**: Firebase Firestore or other database for user data, workflow persistence, and session management
4. **Drag-and-Drop Library**: Libraries like interact.js, dragula, or react-beautiful-dnd (if migrating to React)
5. **Canvas Rendering**: Three.js for the animated background (already integrated in index.html)

### Design Considerations

Future improvements for the application:
- **User data persistence**: Store user preferences and workflows in Firebase Firestore
- **API layer** to connect frontend to AI services
- **State management** solution for the drag-and-drop GUI builder
- **WebSocket or polling** for real-time AI responses
- **Multiple provider handling**: Support users linking multiple authentication providers
- **Fallback avatar system**: Local avatar caching in case DiceBear API is unavailable