# Smol Developer - Feature: Self-Correction/Debugging Loop

This document describes the "Self-Correction/Debugging Loop" functionality for `smol_dev`. This feature enables `smol_dev` to automatically execute its generated code, capture errors or test failures, and use this feedback to iteratively refine the code.

## 1. Core Concept & User Benefit

*   **Concept:** Create an automated loop: Code Generation -> Execution/Testing -> Error/Feedback Capture -> Analysis -> Re-Prompting/Re-Generation -> Repeat. This allows `smol_dev` to autonomously debug and improve its output.
*   **User Benefit:**
    *   **Increased Code Quality:** Automatically identifies and attempts to fix bugs.
    *   **Reduced Debugging Time:** Offloads initial debugging cycles from the user.
    *   **Higher Autonomy:** Makes `smol_dev` a more independent code generation tool.
    *   **Faster Prototyping:** Accelerates the path to a working version.

## 2. High-Level Workflow

1.  **Initial Code Generation:** `smol_dev` generates code from the user's prompt.
2.  **Define Execution/Test Goal:** Determine success criteria (e.g., script runs error-free, web server health check passes, unit tests pass).
3.  **Execution Attempt:** Run the generated code or tests in a controlled environment.
4.  **Error/Feedback Capture:** Collect `stdout`, `stderr`, exit codes, and structured test results if execution fails or tests don't pass.
5.  **Analysis & Root Cause Identification:** An LLM analyzes error messages, stack traces, and relevant code to understand the error and hypothesize a fix.
6.  **Re-Prompting/Re-Generation Strategy:** Based on analysis, decide how to modify the prompt or instruct the LLM for the next generation pass (e.g., refine prompt, instruct LLM to fix specific error, target specific files).
7.  **Code Re-Generation:** `smol_dev` invokes the LLM with the modified instructions.
8.  **Loop/Exit:** Return to Execution Attempt (Step 3) or exit if the goal is met, max attempts are reached, or the process is stuck.
9.  **Report to User:** Summarize the process, fixes applied, and final code state.

## 3. Execution Environment

*   **Sandboxing (Essential):**
    *   **Docker Containers:** Preferred for strong isolation (filesystem, network, processes) and resource control. `smol_dev` might generate or use predefined Dockerfiles.
    *   **Restricted Local Shell:** Simpler but less secure; requires careful permission management.
*   **Handling Project Types:**
    *   **Scripts:** Direct execution (e.g., `python script.py`).
    *   **Web Servers:** Start server, then use a utility (e.g., `curl`) for a health check; ensure graceful shutdown.
    *   **CLI Tools:** Execute with sample arguments.
    *   **Libraries:** Requires generating and running test code.
*   **Security Considerations:**
    *   **Untrusted Code:** Generated code can have unintended or malicious behavior.
    *   **Resource Limits:** Enforce strict CPU, memory, time, and network limits (Docker excels here).
    *   **Filesystem/Network Access:** Restrict writes to temporary directories; deny outbound network access by default or allow only specific ports for web servers.
    *   **User Opt-In & Warnings:** Feature must be opt-in with clear warnings about execution risks.

## 4. Error/Feedback Capture

*   **Standard Streams & Exit Codes:** Capture `stdout`, `stderr`, and process exit codes.
*   **Structured Test Results:** If tests are run (e.g., `pytest`, `Jest`), parse machine-readable output (JUnit XML, JSON).
*   **Timeouts:** Implement to catch hung processes or infinite loops.

## 5. Analysis & Root Cause Identification

*   **LLM as Analyzer:**
    *   **Input:** Error messages, stack traces, relevant code snippets, original prompt context, previous fix attempts.
    *   **Prompting:** "Analyze this [language] error and stack trace. Identify the root cause in the code. Explain and suggest a specific fix."
*   **Distinguishing Error Types:**
    *   **Syntax Errors:** LLM corrects based on compiler/interpreter messages.
    *   **Runtime Errors:** (e.g., `NullPointerException`, `TypeError`). Requires deeper code logic understanding.
    *   **Logical Flaws:** Code runs but produces incorrect results/fails tests. Hardest to diagnose; may involve comparing actual vs. expected output.
*   **Heuristics:** Regex to extract key info from errors (line numbers, variable names) to aid LLM focus.

## 6. Re-Prompting/Re-Generation Strategy

*   **Feedback to LLM:** "Previous code had error [X]. Regenerate [file/function] to fix it while achieving [goal]. Specifically, [suggestion from analysis]."
*   **Scope:** Prefer targeted regeneration (function/file) over broad changes.
*   **Avoiding Infinite Loops:**
    *   **Max Attempts:** Limit iterations (e.g., 3-5).
    *   **Error Repetition/No Progress Detection:** Halt if stuck on the same error or not converging.
    *   **User Intervention Point:** Prompt user for help after several failed attempts.

## 7. User Interaction & Control

*   **Autonomy Level (Configurable):** Fully autonomous, confirm each fix, or report-only.
*   **Transparency:** Log all steps, show code diffs for fixes, distinguish user-prompted vs. LLM-fixed code.
*   **Manual Override:** Allow users to interrupt the loop.

## 8. Key Components/Modules

*   **`CodeExecutor`:** Manages sandboxed execution environment, runs code/tests, captures output.
*   **`OutputParser`:** Converts raw output into structured error/test data.
*   **`ErrorAnalyzer (LLM-based)`:** Diagnoses errors using LLM, suggests fixes.
*   **`PromptUpdater` / `PromptEngine`:** Modifies prompts for LLM based on analysis.
*   **`IterationManager`:** Controls the loop, manages state, orchestrates components.

## 9. Challenges

*   **Error Analysis Complexity:** True root cause analysis is AI-hard.
*   **Safe Code Execution:** Requires robust sandboxing.
*   **Defining "Correctness":** Beyond "no crashes"; functional requirements need to be met (often via tests).
*   **State Management:** Maintaining context across LLM iterations.
*   **Scalability:** More difficult for large, multi-component applications.
*   **Non-Deterministic LLMs:** Variability in fixes.
*   **Environment Flakiness:** Misinterpreting environment issues as code bugs.
*   **Cost/Latency:** Multiple LLM calls can be slow and expensive.

## 10. Integration with `smol_dev`

*   **CLI Flag/Mode:** `smol_dev my_prompt.md --self-correct` or `smol_dev develop my_prompt.md`.
*   **Configuration:** Max iterations, default environment, confirmation levels.
*   **Workflow:** Invoked after initial code generation. Uses generated tests as primary execution targets.
*   **Output:** Reports on iterations, errors, fixes, and final code status.

The Self-Correction/Debugging Loop is an advanced feature aiming to make `smol_dev` significantly more autonomous and effective. Implementation should prioritize safety and start with simpler error types and execution scenarios.
