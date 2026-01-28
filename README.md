# 🤖 OWASP AIBOM Generator

This is the official GitHub repository for the **OWASP AIBOM Generator** — an open-source tool for generating **AI Bills of Materials (AIBOMs)** in [CycloneDX](https://cyclonedx.org) format.  
The tool is also listed in the official **[CycloneDX Tool Center](https://cyclonedx.org/tool-center/)**.

🚀 **Try the tool live:**  
👉 https://owasp-genai-aibom.org  
🔖 Bookmark and share: https://owasp-genai-aibom.org 

🌐 OWASP AIBOM Initiative: [genai.owasp.org](https://genai.owasp.org/)

> This initiative is about making AI transparency practical. The OWASP AIBOM Generator, running under the OWASP GenAI Security Project, is focused on helping organizations understand what’s actually inside AI models and systems, starting with open models on Hugging Face.
> Join OWASP GenAI Security Project - AIBOM Initiative to contribute.

---

## 📦 What It Does

- Extracts metadata from models hosted on Hugging Face 🤗  
- Generates an **AIBOM** (AI Bill of Materials) in CycloneDX 1.6 JSON format  
- Calculates **AIBOM completeness scoring** with recommendations  
- Supports metadata extraction from model cards, configurations, and repository files  

---

## 🛠 Features

- Human-readable AIBOM viewer  
- JSON download & **HTML Reports**  
- Completeness scoring & improvement tips  
- **Data Integrity**: Full commit hash verification & Manufacturer/Supplier mapping
- Standards-aligned generation (CycloneDX 1.6, compatible with SPDX AI Profile)
- **Schema Validation**: Built-in validation against CycloneDX schemas
- **License Normalization**: SPDX-compliant license IDs

---

## � Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Web Application
Start the local server at `http://localhost:8000`:
```bash
python3 -m src.main
```

### 3. Run via CLI
Generate an AIBOM for a Hugging Face model directly from your terminal:
```bash
python3 -m src.cli google-bert/bert-base-uncased
```
*   Metrics and SBOMs are saved to the `sboms/` directory.

---

## �🐞 Found a Bug or Have an Improvement Request?

We welcome contributions and feedback.

➡ **Log an issue:**  
https://github.com/GenAI-Security-Project/aibom-generator/issues

---

## 📄 License

This project is open-source and available under the [Apache 2.0 License](LICENSE).
