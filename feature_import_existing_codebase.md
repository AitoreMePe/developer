# Smol Developer - Feature: Importing Existing Codebases (Reverse Engineering `prompt.md`)

This document describes the functionality for allowing `smol_dev` to analyze an existing directory of code and generate a `prompt.md` file. This generated prompt aims to reproduce a similar codebase when fed back into `smol_dev`.

## 1. Core Concept & Goal

*   **Concept:** Enable `smol_dev` to parse a directory of existing source code and automatically generate a comprehensive `prompt.md` file.
*   **Goal:**
    *   **Onboard Existing Projects:** Facilitate the use of `smol_dev` for ongoing development, refactoring, or documentation of existing projects.
    *   **Learn Prompting:** Help users understand effective `smol_dev` prompting by seeing how their code is deconstructed.
    *   **Template Generation:** Use prompts generated from existing projects as starting points for new, similar projects.
    *   **Codebase Summarization:** Provide a high-level, structured overview of an existing codebase.
*   **Problem Solved:** Reduces manual effort in creating initial prompts for existing projects and lowers the adoption barrier for `smol_dev` on established codebases.

## 2. High-Level Approach

1.  **Codebase Ingestion:** User specifies a target directory.
2.  **File Inventory & Structure Analysis:** Scan for relevant files and map the directory structure.
3.  **Dependency Analysis:** Identify external libraries (e.g., from `requirements.txt`, `package.json`) and internal module dependencies (imports, calls).
4.  **Code Parsing & Abstraction (Per File):** Convert code to ASTs to extract key elements (functions, classes, comments).
5.  **Content Summarization & Intent Inference (Per File/Module):** Use LLMs and heuristics to summarize each file's purpose and infer high-level intent.
6.  **Global Architecture Synthesis:** Analyze inter-component relationships to describe the overall architecture.
7.  **Prompt Assembly:** Structure the collected information into `prompt.md` format, including app description, features, technologies, file structure, and individual file summaries.

## 3. Key Challenges

*   **Code Understanding:**
    *   Distilling complex or poorly documented code into concise, accurate natural language.
    *   Balancing conciseness with the completeness needed for replication.
*   **Identifying Intent:**
    *   Inferring original developer goals beyond literal implementation details.
    *   Handling dead code or legacy constructs.
*   **Handling Diverse Codebases:**
    *   Supporting multiple languages, frameworks, and unconventional coding styles.
    *   Parsing complex build systems and configurations.
*   **Replicability:**
    *   The process is a "lossy compression"; perfect replication is unlikely.
    *   The goal is functional and structural similarity, not a line-for-line match.
    *   Effectiveness will vary with codebase complexity and quality.

## 4. Proposed Workflow for the User

1.  **Initiate Import:** User runs a CLI command (e.g., `smol_dev import --code_dir ./my_project --output_prompt_file ./generated_prompt.md`).
2.  **Analysis Phase:** `smol_dev` analyzes the code, showing progress.
3.  **Review Generated Prompt:** `smol_dev` outputs the `generated_prompt.md`.
4.  **Inspect and Refine:** User reviews and manually refines the generated prompt (correcting interpretations, adding context).
5.  **Test Prompt (Recommended):** User feeds the refined prompt back into `smol_dev` to assess reproduction quality.
6.  **Iterate:** Further refine the prompt or use it as a basis for new development.

## 5. Key Components/Modules

*   **DirectoryScanner:** Traverses directories, identifies files.
*   **DependencyExtractor:** Parses dependency files (e.g., `requirements.txt`) and analyzes internal imports.
*   **LanguageASTParser (Pluggable):** Collection of AST parsers (e.g., Python `ast`, `tree-sitter` for others).
*   **CodeSummarizer (LLM-based):** Uses LLMs to generate natural language summaries from code/ASTs.
*   **ArchitectureAnalyzer:** Infers overall architecture from summaries and dependencies.
*   **PromptFormatter:** Assembles information into the `prompt.md` structure.

## 6. Output: `prompt.md` Structure

The generated `prompt.md` would mirror the manual format:

```markdown
# App Description
[LLM-generated summary of the application's overall purpose and architecture.]

## Key Features
* [Inferred Feature 1 from code analysis]
* [Inferred Feature 2 from code analysis]

## Technologies
* **Programming Languages:** [e.g., Python, JavaScript]
* **Key Frameworks/Libraries:** [e.g., Flask, React]
* **Database:** [e.g., PostgreSQL]

## File Structure & Plan
[High-level description of file organization.]

### `/path/to/file1.py`
[LLM-generated summary of file1.py's purpose and key functionalities.]
[Optional: High-level pseudocode or list of key functions/classes if essential.]

### `/path/to/module/file2.js`
[LLM-generated summary of file2.js's purpose and key functionalities.]

---
(Repeat for all significant files)
---

## Shared Dependencies / Important Notes
* [Notes on environment variables, external service dependencies, etc.]
```
Key aspects include file-specific summaries, abstraction over raw code, and inferred relationships.

## 7. Potential Technologies/Tools

*   **AST Parsers:** Python `ast` module, `tree-sitter` (for multiple languages like JS, Java, Go, etc.).
*   **Large Language Models (LLMs):** OpenAI GPT series, Anthropic Claude, Google Gemini, or open-source alternatives for code summarization and NLU/NLG.
*   **Graph Analysis Tools:** Libraries like `NetworkX` (Python) for complex dependency modeling.
*   **Regular Expressions & Heuristics:** For simpler parsing tasks.

## 8. Integration with `smol_dev`

*   **New CLI Command:** `smol_dev import` or `smol_dev reverse-engineer` with options for input directory, output file, and other parameters (e.g., `--ignore-dirs`, `--include-extensions`).
*   **Library Function:** Expose as a Python function: `smol_dev.reverse_engineer_prompt(code_directory, ...)`.
*   **Core Generation Unchanged:** The core `smol_dev` logic for consuming `prompt.md` remains largely the same.

This feature aims to significantly enhance `smol_dev`'s utility by bridging the gap between existing codebases and prompt-driven development. While challenging, its successful implementation would offer substantial benefits to users.
