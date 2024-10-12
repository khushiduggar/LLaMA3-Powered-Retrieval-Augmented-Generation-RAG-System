# LLaMA3-Powered-Retrieval-Augmented-Generation-RAG-System
This project implements a Retrieval-Augmented Generation (RAG) system powered by LLaMA3. It enables efficient querying of multiple documents(pdfs, excel sheets, ppts, and word documents) The above model works for pdfs and excel sheets to generate accurate, context-aware responses(sample pdf has been attached for reference). Designed for offline use, it ensures data privacy while providing robust natural language understanding and generation capabilities.

# KEY BENEFITS FOR THE COMPANY:

Efficient Information Retrieval:

Organizations can integrate this RAG system to extract insights from large document collections quickly, enabling faster decision-making and knowledge discovery.
Whether it's internal policy documents, research papers, or product manuals, the system can retrieve relevant sections and provide precise answers, saving time and resources.

Enhanced Productivity:

Employees no longer need to manually search through numerous documents. By leveraging the LLaMA3 model's capabilities, they can focus on higher-value tasks, improving operational efficiency.
Supports scalability by allowing large datasets and numerous documents to be processed without requiring cloud services or complex infrastructure.

Offline and Secure:

Since the model works entirely offline, sensitive corporate data never leaves the company’s environment. This guarantees data privacy and ensures compliance with corporate security policies and regulatory standards (e.g., GDPR, HIPAA).
Ideal for industries where strict data privacy is required, such as healthcare, finance, legal, and government sectors.
Customizable and Adaptable:

The system can be easily tailored to suit specific use cases, enabling integration with internal knowledge bases or other document formats beyond PDFs and excel sheets.
This flexibility allows businesses to fine-tune the system to meet domain-specific requirements, creating a specialized tool for their needs.

Anaconda Environment Packaging:

The project leverages Anaconda, making it easy to deploy and manage dependencies in a controlled environment. The provided env.yml file ensures that all necessary packages are installed securely and consistently across multiple systems.
Companies can package the Anaconda environment and distribute it across internal systems without relying on internet access, maintaining control over software versions and security updates.

No External API Calls:

The system is designed to operate without the need for external APIs or third-party services, further enhancing data security. All data processing and querying occur locally, minimizing the exposure of confidential information.
Secure Document Processing:

All documents remain within the local file system, and the system only extracts necessary text for processing. It avoids storing or transmitting sensitive data beyond what is needed for answering queries.

How to Setup?

Setup the Environment:

Install the required dependencies using the provided env.yml file:

(Anaconda Prompt)

conda env create -f env.yml

conda activate testenv

python -m spacy download en_core_web_sm

This will set up all necessary libraries, including:

pdfplumber: for PDF text extraction.

spaCy: for natural language processing and tokenization.

scikit-learn: for additional data processing and analysis.

Note: Ollama has to be downloaded locally into the system and the llama3 model can be pulled into the terminal or can be downloaded externally as well, as per convenience

Run the System:

Place all relevant files in the same directory.
Execute the script:

bash

Copy code

python rag100.py

The system will allow you to input queries and retrieve relevant responses based on the content of the provided documents.
How Companies Can Use This System

# Why Use LLaMA3 for Retrieval-Augmented Generation?

LLaMA3 offers state-of-the-art language generation, capable of understanding complex queries and generating contextually relevant answers from unstructured text data.

Unlike traditional keyword-based search engines, this model understands the context behind user queries, offering more precise and insightful responses.

File Structure

rag100.py: Main script for the RAG system.

env.yml: Anaconda environment file listing all dependencies.

/data: Directory where the files should be stored for querying.
