# Smol Developer - Feature: Enhanced `shared_dependencies.md` Management

This document describes enhancements to the `shared_dependencies.md` file (or an equivalent mechanism) to improve coherence, robustness, and user control in `smol_dev`-generated projects.

## 1. Core Concept & Goal

*   **Current Purpose:** `shared_dependencies.md` acts as a central reference for information (architecture, data structures, notes) that needs to be consistent across multiple generated files, aiming to ensure interoperability.
*   **Current Limitations (Implied):** Potential for vagueness in Markdown, inconsistent LLM adherence, manual upkeep challenges, scalability issues for large projects, and possibly limited scope of shared information.
*   **Goal of Enhancements:**
    *   Improve overall code quality and inter-module coherence.
    *   Reduce integration errors stemming from inconsistencies.
    *   Enable generation of more complex and larger applications.
    *   Increase the reliability and predictability of LLM code generation.
    *   Offer better developer control over cross-cutting concerns and shared logic.

## 2. Proposed Enhancements to Content & Structure

*   **Expanded Information Scope:**
    *   **Global Constants & Enums:** Centralized definitions.
    *   **Core Data Structures/Models:** Detailed schemas (e.g., for DTOs, ORM models).
    *   **Key Function/Method Signatures:** Cross-module API contracts.
    *   **API Endpoint Definitions:** For services (paths, methods, request/response formats).
    *   **Event Definitions:** For event-driven systems (names, payloads).
    *   **State Management Strategies:** For frontend or stateful applications.
    *   **Error Handling Conventions:** Common error types/codes.
    *   **Key Configuration Settings:** Names and expected types.
    *   **Database Schema Snippets:** Key table definitions or ORM outlines.
*   **Format Considerations:**
    *   **Markdown (Current):** Human-readable, LLM-friendly but prone to ambiguity and inconsistent adherence.
    *   **YAML (Recommended):** Good balance of human readability and structure, supports comments, suitable for config-like data.
    *   **JSON:** Highly structured, unambiguous parsing, good for data interchange, but less human-friendly.
    *   **Hybrid Approach:** Use YAML as primary, potentially embedding JSON for strict schemas or linking to Markdown for prose descriptions. Critical definitions should be in a strictly parsable format.

## 3. Generation Process

*   **Current (Likely):** Initial LLM call based on main prompt generates `shared_dependencies.md`.
*   **Improvements:**
    *   **Dedicated Prompting:** Use a specific, detailed prompt template solely for generating this artifact, guiding the LLM on categories of shared information.
    *   **Iterative LLM Refinement:** Use subsequent LLM calls to review, critique, and augment the initial shared dependency draft.
    *   **Two-Pass Generation:** 1) High-level plan from main prompt. 2) Dedicated LLM call creates detailed shared dependencies from this plan.

## 4. Utilization Process

*   **Current (Likely):** Entire `shared_dependencies.md` included in context for each file generation.
*   **Improvements:**
    *   **Strict Adherence Instructions:** Explicitly instruct the file-generating LLM that shared dependency definitions are mandatory.
    *   **Structured Injection:** Parse structured shared dependencies (YAML/JSON) and inject only relevant parts for the specific file being generated.
    *   **Chunking for Large Projects:** Develop a system to identify and provide only relevant sections of large shared dependency files to the LLM to respect context limits.
    *   **Post-Generation Validation:** Check generated files for adherence to shared definitions.

## 5. Interactive Management (Linking to Interactive Prompt Interface)

*   **User Review & Editing:** The interactive UI should display parsed shared dependencies, allowing users to view, add, delete, or modify entries.
*   **Guided Edits:**
    *   Use forms or dedicated UI for structured data entry (e.g., schemas, API contracts).
    *   Provide real-time syntax and semantic validation.
    *   Offer suggestions, autocompletion, and templates.
    *   (Advanced) Basic impact analysis of changes.

## 6. Validation & Consistency Checking

*   **Against Main Prompt:** LLM-based assessment of whether shared dependencies accurately reflect main prompt requirements.
*   **Against Generated Code:**
    *   **Static Analysis:** Linters/analyzers to check if code uses interfaces, data structures, etc., as defined.
    *   **LLM-based Validation:** Ask an LLM if generated code adheres to specified shared definitions.
*   **Internal Consistency:** Schema validation for YAML/JSON; check for undefined internal references.

## 7. Key Components/Modules

*   **`SharedDepsGenerator`:** Improved LLM prompting/logic for initial generation.
*   **`SharedDepsParser`:** Parses various formats (Markdown, YAML, JSON) into an internal model.
*   **`SharedDepsValidator`:** Validates syntax, semantics, and consistency.
*   **File Generation Logic Integration:** Ensures relevant shared info is passed to and emphasized for the LLM.
*   **Interactive UI Component:** For viewing/editing shared dependencies.

## 8. Challenges

*   **Defining Appropriate Detail Level:** Balancing too little vs. too much information.
*   **Ensuring LLM Adherence:** Forcing strict compliance without stifling useful generation.
*   **Semantic Validation Complexity:** Difficult to validate beyond syntax.
*   **Synchronization:** Handling conflicts if users directly edit code that's also defined in shared dependencies.
*   **User Effort:** Must simplify, not complicate, the user's task.

## 9. Integration with `smol_dev`

*   **Workflow Timing:** Generated/updated after initial main prompt processing, before individual file generation. Validated at various stages.
*   **Iterative Refinement:** Errors found during self-correction could lead to updates in shared dependencies.
*   **Configuration:** User might choose format (YAML/JSON) or detail level.

Enhanced management of shared dependencies is crucial for `smol_dev` to generate more complex, robust, and coherent applications, making it a more powerful and reliable tool. A shift to structured formats, interactive editing, and robust validation are key pillars of this enhancement.
