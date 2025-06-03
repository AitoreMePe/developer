# LLM Studio Prompt Templates

This file contains a collection of prompt templates for common tasks that can be used with LLM Studio or similar large language model interfaces.

## 1. Code Generation

### 1.1. Generate a Python function

```
Generate a Python function that takes [input parameters] and returns [expected output].

Function description: [Detailed description of what the function should do, including any specific algorithms or libraries to use.]

Example usage:
[Optional: Provide an example of how the function should be called and what the output should be.]
```

### 1.2. Generate HTML/CSS for a component

```
Create the HTML and CSS code for a [component name, e.g., responsive navigation bar, product card].

Requirements:
- [List specific features, e.g., must be responsive, include a logo section, have dropdown menus]
- [Styling preferences, e.g., modern look, use a specific color palette]
```

### 1.3. Generate a SQL Query

```
Write a SQL query to [desired outcome, e.g., select all users who signed up in the last month and have made at least one purchase].

Database schema:
[Provide relevant table names, column names, and relationships, e.g.,
Users(user_id, name, signup_date)
Orders(order_id, user_id, purchase_date, amount)
]
```

## 2. Text Summarization

### 2.1. Summarize an article

```
Summarize the following text into [number] sentences/paragraphs, focusing on the key points:

[Paste article text here]
```

### 2.2. Extract key information

```
Extract the key information (e.g., names, dates, locations, main topics) from the following text:

[Paste text here]
```

### 2.3. Bullet-point summary

```
Provide a bullet-point summary of the main arguments/findings in the text below:

[Paste text here]
```

## 3. Question Answering (with Context)

### 3.1. Answer questions based on a document

```
Based on the provided text, answer the following question:

Context:
---
[Paste context/document text here]
---

Question: [Your question here]

Answer:
```

### 3.2. Factual verification

```
Please verify the following statement based on the provided context. Explain your reasoning.

Context:
---
[Paste context/document text here]
---

Statement: [Statement to verify]

Verification and Explanation:
```

## 4. Creative Writing

### 4.1. Story generation

```
Write a short story (around [word count] words) based on the following prompt:

Prompt: [e.g., A detective in a futuristic city investigates a mysterious signal from an abandoned part of town.]

Include the following elements/themes: [Optional: e.g., betrayal, advanced technology, a surprising twist]
```

### 4.2. Poem generation

```
Write a poem in the style of [poet's name, e.g., Edgar Allan Poe, Maya Angelou] about [subject].

The poem should be [number] stanzas long and follow a [rhyme scheme, e.g., AABB, ABAB] if applicable.
```

### 4.3. Dialogue writing

```
Write a dialogue between two characters: [Character A description] and [Character B description].

Scenario: [Describe the situation or topic they are discussing, e.g., They are arguing about a recently discovered artifact.]

The dialogue should reveal [e.g., Character A's hidden fear and Character B's ambition].
```

## 5. General Purpose / Utility

### 5.1. Brainstorming ideas

```
Brainstorm a list of ideas for [topic, e.g., a new mobile application, a marketing campaign for a sustainable product].

Consider the following aspects: [Optional: e.g., target audience, potential challenges, unique selling points]
```

### 5.2. Rephrasing text

```
Rephrase the following text to be more [desired tone, e.g., formal, concise, persuasive, simpler]:

Original text:
[Paste text here]

Rephrased text:
```

### 5.3. Classification

```
Classify the following text into one of these categories: [Category A, Category B, Category C, ...]. Explain your reasoning.

Text:
[Paste text here]

Classification and Reasoning:
```
