# Document Portal Analysis with RAG

## Business Use Case

In a company, when purchasing products from a third-party vendor, the vendor often provides invoices or reports related to electronic devices from various chip makers worldwide. Instead of manually reviewing these invoices or reports, we need a streamlined solution for analyzing and comparing them. The **Document Portal Analysis** application aims to build a portal that allows for comparison, analysis, and interactive communication with these documents.

### Features

The application provides four core services for document management:

1. **Document Analysis**  
   This service analyzes any document (e.g., PDFs) uploaded by the user. Upon uploading, the system performs a detailed analysis of the document's contents, providing insights and relevant information.

2. **Document Compare**  
   Users can upload two different PDFs, and the system will compare them side-by-side, highlighting the differences. This service is useful for tracking variations in invoices or reports from multiple vendors.

3. **Single Document Chat**  
   This feature allows users to interact with a single document by uploading a PDF and asking questions related to its content. The system will provide relevant responses based on the document's text.

4. **Multi-Document Chat**  
   Users can upload multiple PDFs and interact with them simultaneously. This service enables querying across multiple documents, offering a holistic view of the provided data.

## Technologies Used
- Python
- Langchain
- FastAPI
- Streamlit
- Google Embeddings Model
- FAISS
- Groq LLM Models
- Google Gemini LLM Model
- Docker
- CI/CD Pipeline with Github Actions
- AWS (ECR, Fargate, ECS, Secret Manager)

## Installation

1. **Clone the repository**:

   git clone https://github.com/Anuragreddy-Naredla/document_portal_analysis_with_adv_RAG.git

2. **Create the virtual environment**:

    conda create -p env python=3.10 -y

3. **Activate the virtual environment**:

    conda activate env

4. **Install the required dependencies**:

    pip install -r requirements.txt

**Generate API Keys**

1. **Create a .env file** in your current folder.

2. **Groq API Key:**

    * Go to the Groq Console(https://console.groq.com/keys)

    * Go to the API Keys tab.

    * Click on "Create API Key" and copy the key.

    * Paste the key in the .env file:
        GROQ_API_KEY="YOUR_API_KEY_HERE"
3. **Google API Key:**
    * Go to Google AI Studio(https://aistudio.google.com/prompts/new_chat)
    * Create an API key.
    * Paste the key in the .env file:
        GOOGLE_API_KEY="YOUR_API_KEY_HERE"

**Deployment Steps**
**Configure AWS Services**

1. **Configure ECR (Elastic Container Registry):**

    * Search for "ECR" and click on "Elastic Container Registry".

    * Select the region (e.g., ap-southeast-2).

    * Click on "Create Repository".

    * Name the repository (e.g., documentportalsystem), matching the name in the aws.yaml file under ECR_REPOSITORY.

    * Enable "Scan on push" under Image Scanning Settings.

    * Copy the "URI" and save it for later.

2. **Configure IAM User:**

    * Search for "ECR" and click on "Elastic Container Registry".
    * Go to the "IAM" service, then click on "Users" under "Access Management".

    * Click on "Create user", assign a name, and attach the following policies:

        * AmazonEC2ContainerRegistryFullAccess

        * AmazonECS_FullAccess

        * AmazonS3FullAccess

        * SecretsManagerReadWrite

        * CloudWatchLogsFullAccess

    * Copy the "ARN" under "Summary" after creating the user.

    * Create access keys for CLI access. Store the "Access key" and "Secret access key" securely.

    * Add these credentials to GitHub Secrets for CI/CD integration:

        * AWS_ACCESS_KEY_ID

        * AWS_SECRET_ACCESS_KEY

3. **Configure ECS (Elastic Container Service):**

    * Search for ECS and click on "Create Cluster".

    * Use the cluster name from the aws.yaml file.

    * Choose "AWS Fargate (serverless)" for the launch type.

    * Create the task definition and configure with the task_definition.json file.

    * Deploy the task.

4. **Configure AWS Secret Manager:**

    * Search for "Secret Manager" and click on "Store a new secret".

    * Add the following keys:

        * GROQ_API_KEY

        * GOOGLE_API_KEY

    * Store the secret.

5. **Execute the Application:**

    * Run the application at least once to generate the ecsTaskExecutionRole.

    * Add the required policies (from incline_policy.json) to the IAM role for access.

6. **Security Groups and Secret Manager:**

    * Set up security groups from EC2.

    * Update the task_definition.json with the secret values in SECRET MANAGER:

        "Secrets": [
                {
                    "Name": "GROQ_API_KEY",
                    "ValueFrom": "YOUR_SECRET_MANAGER_KEY"
                },
                {
                    "Name": "GOOGLE_API_KEY",
                    "ValueFrom": "YOUR_SECRET_MANAGER_KEY"
                }
                ]

**Accessing the Application**

    1. Go to the ECS service and open the cluster.

    2. Go to "Tasks" and click on the most recent task.

    3. Under "Configuration", copy the "Public IP".

    4. Open a browser and navigate to http://<Public_IP>:8080.

**Cleanup**
 * After completing the project, delete the AWS services (ECR, ECS) to avoid ongoing charges.

 ![alt text](doc_analysis.PNG)

 ![alt text](doc_chat.PNG)

![alt text](doc_compare.PNG)