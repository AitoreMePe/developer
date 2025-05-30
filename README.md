# Smol Developer - Interactive Prompt Refinement Interface

This document outlines the concept and design for an Interactive Prompt Refinement Interface for `smol_dev`. This interface aims to enhance the user experience of creating and iterating on prompts used to guide `smol_dev` in code generation.

## 1. Core Idea

The core idea is to replace or augment the static `prompt.md` file with a dynamic, guided, and user-friendly interface. This interface will help users construct well-structured and effective prompts, leading to better code generation results from `smol_dev`.

**User Benefits:**

*   **Improved Prompt Quality:** Structured input and real-time feedback lead to clearer, more comprehensive, and less ambiguous prompts.
*   **Reduced Errors & Faster Iteration:** Early validation and previews catch issues before full code generation, allowing for quicker refinement cycles.
*   **Easier Onboarding:** Lowers the barrier to entry for new `smol_dev` users by simplifying prompt creation.
*   **Enhanced Control & Predictability:** Users gain finer-grained control over the input, leading to more predictable outputs.

## 2. Interface Modality

While several modalities exist (Web UI, CLI, VS Code Extension), a **VS Code Extension** is proposed as the primary modality.

*   **Pros of VS Code Extension:**
    *   **Integrated Developer Experience:** Lives directly within the developer's primary tool.
    *   **Excellent Filesystem Integration:** Seamless access to project workspace and files.
    *   **Rich UI Capabilities:** Can leverage VS Code's Webview API for custom UIs (HTML, CSS, JS).
    *   **Context-Awareness:** Can use information from the current workspace.
    *   **Leverage Existing VS Code Features:** Utilizes built-in notifications, input boxes, tree views, etc.

*   **Cons of VS Code Extension:**
    *   **IDE-Specific:** Benefits only VS Code users.
    *   **Development Learning Curve:** Requires knowledge of VS Code extension APIs.

A Web UI could be a future option for broader accessibility, and a CLI could serve as a supplementary tool.

## 3. Key Features

*   **Structured Prompt Building:**
    *   **Dedicated Sections:** UI with distinct input areas for:
        *   Project Overview/Goal
        *   Key Features/User Stories
        *   Target Technologies/Frameworks
        *   Non-Functional Requirements (performance, security, logging)
        *   Existing Code Context (optional)
        *   File/Folder Structure Preferences (optional)
    *   **Templates/Examples:** Predefined templates for common application types (e.g., "Web API," "CLI Tool").
    *   **Guided Questions:** Placeholder text and guiding questions to encourage comprehensive input.

*   **Real-time Feedback/Validation:**
    *   **Clarity Suggestions:** NLP-based (or heuristic-based) suggestions for vague phrases.
    *   **Ambiguity Warnings:** Identification of conflicting or underspecified requirements.
    *   **Complexity Estimation (Basic):** Rough estimate of project scope based on prompt content.
    *   **Keyword Highlighting:** Highlighting of recognized technologies, libraries, etc.
    *   **Missing Information Prompts:** Reminders for empty crucial sections.

*   **File Structure Preview:**
    *   **Dynamic Tree View:** Real-time preview of the predicted file/folder structure based on the current prompt.
    *   **Heuristics-Based:** Predictions based on common project structures for specified technologies and features.

*   **Dependency Suggestion:**
    *   **Keyword/Feature-Based:** Suggests potential libraries/dependencies based on prompt content (e.g., "Python" + "database" -> "SQLAlchemy").
    *   **Opt-In:** Suggestions are clearly marked and easily accepted/ignored.

*   **Prompt History and Versioning:**
    *   **Automatic Saving:** Prompts saved automatically.
    *   **Named Snapshots:** Users can save prompt versions with custom names.
    *   **Diff View:** Comparison between different prompt versions.
    *   **Revert Functionality:** Ability to revert to previous snapshots.
    *   **Storage:** Local storage within the VS Code extension (e.g., JSON files).

*   **Integration with `smol_dev` Core:**
    *   **Prompt Generation:** Interface outputs a `prompt.md` file or a structured format `smol_dev` can consume.
    *   **Invoking `smol_dev`:** UI button to trigger the `smol_dev` Python script.
    *   **Output Display:** Show `smol_dev` progress, logs, and potentially file diffs within VS Code.
    *   **Configuration:** UI for `smol_dev` parameters (model, temperature, etc.).

## 4. User Workflow

1.  **Open Interface:** Launch "Smol Dev Prompt Refiner" in VS Code.
2.  **Start/Load Prompt:** Create a new prompt (optionally from a template) or load a saved one.
3.  **Structured Input:** Fill in sections, receiving real-time feedback, file structure previews, and dependency suggestions.
4.  **Iterate & Refine:** Review feedback/previews, adjust the prompt, save snapshots.
5.  **Configure `smol_dev` (Optional):** Set generation parameters.
6.  **Generate Code:** Click "Generate Code" to run `smol_dev` with the current prompt.
7.  **Review Output:** Examine `smol_dev`'s output (logs, generated files) in VS Code.
8.  **Further Refinement:** If needed, return to the prompt refiner, adjust, and regenerate.

## 5. Technology Stack (High-Level - VS Code Extension)

*   **Frontend (Webview UI):**
    *   HTML, CSS, JavaScript/TypeScript
    *   UI Framework (Recommended): React, Vue, or Svelte.
    *   VS Code Webview UI Toolkit.
*   **Backend (Extension Logic):**
    *   TypeScript/JavaScript.
    *   Node.js APIs for VS Code interaction.
*   **Communication with `smol_dev` Python core:**
    *   Child Process (`child_process.spawn`).
    *   Standard I/O or IPC for communication; prompt likely passed via a temporary file.
*   **NLP/Text Analysis (Optional):**
    *   Start with regex/keyword matching; potentially lightweight NLP libraries.
*   **Prompt Storage:**
    *   VS Code Extension Storage API.
    *   Local JSON files for snapshots.

This interface aims to significantly improve the developer experience when working with `smol_dev`, making prompt engineering more intuitive, efficient, and effective.
