# Smol Developer - Feature: Version Control Integration (Automated Branching/Committing)

This document describes the functionality for integrating `smol_dev` with Git to automatically manage branches and commits for generated or modified code.

## 1. Core Concept & User Benefit

*   **Concept:** Seamlessly integrate `smol_dev` with Git to automate versioning of generated code, including branch creation and committing changes.
*   **User Benefit:**
    *   **Traceability & History:** Track `smol_dev`'s changes over time with distinct commits.
    *   **Experimentation & Safety:** Isolate generation attempts in branches, protecting main work.
    *   **Reduced Manual Git Work:** Automate common Git operations.
    *   **Collaboration:** Facilitate review of `smol_dev`-generated changes via shared branches.
    *   **Standard Workflow Integration:** Fits naturally into Git-based development.

## 2. Git Operations to Automate

*   **Pre-check:** Verify it's a Git repository and check for uncommitted user changes.
*   **Branch Creation:** `git checkout -b <branch_name>` (e.g., from current or specified base branch).
*   **Staging Changes:** `git add <specific_files_generated_or_modified>` (preferred) or `git add .` (if confined and respecting `.gitignore`).
*   **Committing Changes:** `git commit -m "<generated_commit_message>"`.
*   **Tagging (Optional):** `git tag <tag_name>` for significant milestones.
*   **Switch Back (Optional):** `git checkout <original_branch_name>` after operations.

## 3. Branching Strategy

*   **Naming Conventions:**
    *   Prefix: `smol_dev/` (e.g., `smol_dev/feature-name`, `smol_dev/generated-YYYYMMDD-HHMMSS`).
    *   Content: Based on prompt keywords, user-supplied name, or timestamp.
    *   Iteration: For self-correction loops, e.g., `smol_dev/feature-x-iteration-3`.
*   **Logic:**
    *   **New Branch Per Run (Default):** Isolates each `smol_dev` invocation.
    *   **Work on Existing `smol_dev` Branch (Optional):** User can specify an existing `smol_dev/` branch to add commits to.

## 4. Commit Message Generation

*   **Goal:** Automatic, meaningful messages.
*   **Methods:**
    *   Summary of the `prompt.md`.
    *   List of key features from a structured prompt.
    *   List of changed/generated files.
    *   LLM-generated summary based on prompt/changes.
    *   Fixed prefix + details (e.g., "smol_dev: Generated user authentication").
    *   Iteration info for self-correction loops.
*   **Style:** Imperative mood (e.g., "Add login endpoint").

## 5. User Interaction & Configuration

*   **Opt-In Feature:** Disabled by default; enabled by CLI flag (e.g., `--git-auto-commit`) or config file.
*   **User Control:**
    *   Allow user to suggest branch name/suffix.
    *   Allow user to provide a custom commit message.
    *   Optional interactive mode for confirmations.
*   **Handling Uncommitted Changes (Pre-run):**
    *   Default: Refuse to run if conflicting uncommitted changes exist.
    *   Force option (risky).
    *   Stash option (advanced, complex).

## 6. Error Handling

*   **Git Not Installed / Not a Repo:** Warn user and disable feature or offer `git init`.
*   **Git Command Failures:** Report Git error to user, halt `smol_dev` Git operations, leave files for manual intervention.
*   **Merge Conflicts:** Handled by standard Git workflow when user merges `smol_dev` branch. Avoid concurrent manual edits on active `smol_dev` branches.

## 7. Key Components/Modules

*   **`GitClient` Wrapper:** Module (using `subprocess` or `GitPython`) for Git commands (`is_repo`, `create_branch`, `add`, `commit`, etc.).
*   **`BranchNameGenerator`:** Creates branch names per strategy.
*   **`CommitMessageGenerator`:** Creates commit messages (possibly LLM-assisted).
*   **`ConfigManager`:** Reads Git integration settings.

## 8. Security/Safety

*   **Respect `.gitignore`:** Crucial when staging files. Prefer adding specific files `smol_dev` touched.
*   **No Sensitive Data in Commits:** `smol_dev` should not introduce sensitive data. User prompt content used in messages is user's responsibility.
*   **Command Injection:** Sanitize user/LLM inputs used in shell commands if not using a Git library.

## 9. Challenges

*   **Useful Commit Messages:** Automatic generation can be hard.
*   **Robust Branching:** Needs to be intuitive and avoid clutter.
*   **Non-Interference:** Must not disrupt user's Git practices.
*   **Idempotency:** Behavior on re-running with the same prompt.

## 10. Integration with `smol_dev`

*   **Workflow Step:** Git operations occur after code generation/modification.
*   **Pre-run Check:** For uncommitted changes.
*   **Post-generation Actions:** Branch, stage, commit message generation, commit, optional tag, optional switch back.
*   **Configuration:** CLI flags (`--git-auto-commit`) or config file.
*   **Self-Correction Loop:** Each iteration could be a commit on a feature branch.

This feature aims to integrate `smol_dev` smoothly into standard Git workflows, enhancing traceability and safety for AI-assisted code generation.
