# Smol Developer - Feature: Automated Dependency Management and Installation

This document outlines the functionality for `smol_dev` to identify, report, and optionally install external dependencies required by the code it generates.

## 1. Core Concept & User Benefit

*   **Concept:** `smol_dev` will automatically detect external libraries/packages needed by the generated code, inform the user, and offer to manage their installation.
*   **User Benefit:**
    *   **Reduced Manual Work:** Eliminates manual scanning of code for imports and package lookup.
    *   **Faster Project Setup:** Streamlines getting a generated project running.
    *   **Improved Accuracy:** Minimizes missed dependencies or incorrect package installations.
    *   **Better Reproducibility:** Facilitates environment recreation through standard dependency files.
    *   **Smoother Onboarding:** Simplifies dependency handling for new users.

## 2. Dependency Identification Methods

*   **Static Analysis of Import/Require Statements:**
    *   **Python:** Parse `import X`, `from X import Y` using `ast`. Differentiate standard library from external packages.
    *   **JavaScript/TypeScript:** Parse `require('X')`, `import Y from 'X'` using tools like `esprima` or `tree-sitter`.
    *   **Java:** Analyze `import` statements; more reliably, parse `pom.xml`/`build.gradle` if `smol_dev` manages them.
    *   **Go:** Analyze `import` paths which map directly to package names.
*   **LLM-based Recognition:**
    *   The code-generating LLM can be prompted to list dependencies it used.
    *   A separate LLM call can analyze code snippets to identify likely library usage.
*   **Heuristics & Common Patterns:** Recognize CLI tools or system utilities.
*   **Language/Package Manager Specificity:**
    *   Maintain internal mappings (e.g., Python module `bs4` to package `beautifulsoup4`).
    *   Employ pluggable analyzers for different languages/package managers.
    *   Prioritize common systems like Python/pip and Node.js/npm initially.

## 3. Reporting Dependencies

*   **Console Output:** Clearly list identified dependencies post-generation, grouped by language/package manager, including potential version suggestions.
*   **Generating/Updating Standard Dependency Files:**
    *   **Python:** Create/update `requirements.txt`.
    *   **Node.js:** Create/update `package.json` (dependencies/devDependencies).
    *   **Java (Maven/Gradle):** Add/update entries in `pom.xml` or `build.gradle`.
    *   **User Confirmation:** Always prompt before modifying existing dependency files.

## 4. Automated Installation (Optional)

*   **User Prompt:** Ask the user if they wish to proceed with installation after reporting.
*   **Command Execution:** If confirmed, run appropriate commands (e.g., `pip install -r requirements.txt`, `npm install`).
*   **Security Implications & Mitigations:**
    *   **Risks:** Arbitrary code execution via installation scripts, malicious packages.
    *   **Mitigations:**
        *   **Explicit User Confirmation:** Primary safeguard; user must agree.
        *   **Clear Reporting:** Show exact commands to be run.
        *   **Trusted Sources:** Use official package repositories.
        *   **Virtual Environments:** Strongly encourage/enforce for Python to contain dependencies.

## 5. User Workflow

1.  User runs `smol_dev` to generate code.
2.  `smol_dev` generates code.
3.  **Dependency Identification:** `smol_dev` analyzes generated code.
4.  **Reporting:**
    *   Lists dependencies in the console.
    *   Prompts to create/update dependency files (e.g., `requirements.txt`); user confirms.
5.  **Optional Installation:**
    *   Asks user if they want to install dependencies.
    *   If yes, shows commands, asks for final confirmation, then executes.
    *   Reports success/failure.

## 6. Key Components/Modules

*   **`DependencyIdentifier` (Abstract):** With language-specific parsers (e.g., `PythonImportParser`, `JavaScriptImportParser`) and an optional `LLMDependencySuggester`.
*   **`DependencyFileGenerator` (Abstract):** With generators like `RequirementsTxtGenerator`, `PackageJsonGenerator`.
*   **`PackageManagerRunner` (Abstract):** With runners like `PipRunner`, `NpmRunner`.
*   **`MappingService`:** Maps import names to package names, potentially with version info.
*   **`VirtualEnvDetector`:** (Python) Checks for active virtual environments.

## 7. Configuration & Customization

*   **Global `smol_dev` Config:** `auto_install_dependencies` (boolean/ask), preferred dependency file format.
*   **Prompt-Level Specification (Advanced):** Allow version hints in the prompt (e.g., "use Flask version 2.x").
*   **Virtual Environments:** For Python, detect and prefer; warn if global. Node.js typically local by default.
*   **Version Conflict Resolution:** `smol_dev` will not attempt to resolve complex conflicts. It will list dependencies (with versions if known) and let the native package manager handle resolution during installation.

## 8. Challenges

*   **Accuracy of Detection:** Standard vs. external libraries, aliased/dynamic imports, mapping usage to correct package names (e.g., `cv2` from `opencv-python`).
*   **Version Conflicts:** Handled by the package manager, not `smol_dev`.
*   **Broad Language/Manager Support:** Requires significant effort; adopt an incremental approach.
*   **Safe Command Execution:** Relies on user consent and awareness.
*   **Build Tools for Compiled Languages:** Modifying files like `pom.xml` is complex; initial support might be reporting only.
*   **Contextual Dependencies:** Differentiating dev vs. runtime dependencies.

## 9. Integration with `smol_dev`

*   **Post-Generation Hook:** Runs after code generation.
*   **CLI Flags:** `--skip-dependency-check`, `--install-deps yes|no|ask`.
*   **Configuration File:** For persistent settings.
*   **LLM Prompt Augmentation:** The main code-generating LLM can be asked to list libraries it used, feeding this information to the dependency identification step.

This feature will make `smol_dev` more practical and user-friendly by automating a crucial but often tedious part of project setup. Prioritizing common ecosystems and ensuring user control over installations will be key to its success.
