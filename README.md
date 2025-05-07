# **Vector Embeddings Application: Architecture & Implementation Guide**  

## **📌 Table of Contents**  
1. [**Overview**](#-overview)  
2. [**Architecture Diagram**](#-architecture-diagram)  
3. [**Key Features**](#-key-features)  
4. [**Prerequisites**](#-prerequisites)  
5. [**Setup & Configuration**](#-setup--configuration)  
   - [OpenAI Model Deployment](#-1-openai-model-deployment)  
   - [Azure PostgreSQL Flexible Server Setup](#-2-azure-postgresql-flexible-server-setup)  
   - [PgAdmin 4 Integration](#-3-pgadmin-4-integration)  
6. [**Application Workflow**](#-application-workflow)  
7. [**Semantic Search Implementation**](#-semantic-search-implementation)  
8. [**Test Results & Evidence**](#-test-results--evidence)  
9. [**Benefits & Use Cases**](#-benefits--use-cases)  
10. [**Troubleshooting**](#-troubleshooting)  

---

## **🌐 Overview**  
This application provides an **application-level solution** for generating, storing, and querying **vector embeddings** using:  
- **OpenAI** (for generating embeddings)  
- **Azure PostgreSQL Flexible Server** (with `pgvector` for vector storage)  
- **FastAPI** (for REST API endpoints)  

The system enables **semantic search**, allowing users to find documents based on **meaning similarity** rather than exact keyword matches.  

---

## **📊 Architecture Diagram**  
*![image](https://github.com/user-attachments/assets/0d7d7224-90b7-4070-a1fd-6e3d9ca40412)*  

### **Key Components:**  
1. **OpenAI API** – Generates embeddings (`text-embedding-3-small`)  
2. **FastAPI Backend** – Processes requests and stores embeddings  
3. **PostgreSQL + pgvector** – Stores and queries vectors efficiently  
4. **PgAdmin 4** – Database management & monitoring  

---

## **✨ Key Features**  
✅ **Automated Embedding Generation** – OpenAI processes text into vectors  
✅ **Scalable Storage** – Azure PostgreSQL with `pgvector` extension  
✅ **Semantic Search** – Find documents by meaning, not just keywords  
✅ **Batch Processing** – Worker script handles large document volumes  
✅ **Hybrid Deployment** – Supports both OpenAI and Azure OpenAI  

---

## **⚙️ Prerequisites**  
Before setup, ensure you have:  
- **Azure Account** (for PostgreSQL & OpenAI)  
- **OpenAI API Key** (or Azure OpenAI endpoint)  
- **Python 3.9+** (with `pip`)  
- **PgAdmin 4** (for DB management)  

---

## **🔧 Setup & Configuration**  

### **1️⃣ OpenAI Model Deployment**  
#### **Steps to Configure OpenAI Embeddings**  
1. **Get API Key**  
   - Visit [OpenAI API Keys]([https://platform.openai.com/account/api-keys](https://ai.azure.com/resource/overview?wsid=/subscriptions/afa8b552-6773-4cb2-bf7e-4e037eb843aa/resourceGroups/vEmbeddings/providers/Microsoft.CognitiveServices/accounts/appEmbeddings&tid=34370bc8-f92f-4979-a403-e166e31bb907))  
   - Create a new key and store it in `.env`:  
     ```env
     AZURE_OPENAI_API_KEY=xxxxxxxxxxxxxxxx
     ```  
2. **Test Embedding Generation**  
   ```python
   from openai import OpenAI
   client = OpenAI(api_key="your-key")
   response = client.embeddings.create(input="test", model="text-embedding-3-small")
   print(response.data[0].embedding[:5])  # First 5 dimensions
   ```
   *![testresultsembeddings](https://github.com/user-attachments/assets/61b8fdcd-adf8-49bd-85a5-d148c04a4614)* 

---

### **2️⃣ Azure PostgreSQL Flexible Server Setup**  
#### **A. Create PostgreSQL Server**  
1. Go to **Azure Portal** → **Create PostgreSQL Flexible Server**  
2. Configure:  
   - **Server Name**: `vector-db-server`  
   - **Admin Username/Password**  
   - **Enable `pgvector` extension**  

#### **B. Connect via PgAdmin 4**  
1. **Add Server** in PgAdmin  
   - Host: `your-server.postgres.database.azure.com`  
   - Port: `5432`  
   - Username: `admin-user`  
   - Password: `your-password`  
2. **Enable `pgvector`**  
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
   *![vectorcreated](https://github.com/user-attachments/assets/44849988-3282-4d21-bfd7-928b99697baa)* 

---

### **3️⃣ Application Deployment**  
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-repo/vector-embeddings.git
   cd vector-embeddings
   ```
2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run FastAPI Server**  
   ```bash
   uvicorn main:app --reload
   ```
4. **Run Worker Script**  
   ```bash
   python worker.py
   ```

---

## **🔄 Application Workflow**  
1. **Document Ingestion**  
   - POST `/documents/` → Stores text in PostgreSQL  
   - Worker generates embeddings via OpenAI  
2. **Semantic Search**  
   - GET `/search/?query=your+text` → Returns similar documents  

*![similarity search](https://github.com/user-attachments/assets/d76eaf2d-cbf2-4a27-a0f1-52ca38c482ce)* 

---

## **🔍 Semantic Search Implementation**  
### **How It Works**  
1. **Query Embedding** – Convert search text into a vector  
2. **Cosine Similarity** – Compare against stored vectors  
3. **Rank Results** – Return most similar documents  

### **Example Query**  
```sql
SELECT 
    id, 
    content,
    embedding <=> '[0.1, 0.2, ...]' AS similarity
FROM documents
ORDER BY similarity
LIMIT 5;
```
*![testresultsembeddings](https://github.com/user-attachments/assets/1549ef24-6283-4b82-a940-96b8890472bc)* 

---

## **📊 Test Results & Evidence**  
### **1. Database Tables**  
| Table | Columns | Sample Data |
|-------|---------|-------------|
| `documents` | `id`, `content` | `"PostgreSQL with pgvector"` |
| `document_embeddings` | `document_id`, `embedding` | `[0.1, -0.3, ...]` (1536D) |

*![testresultsembeddings](https://github.com/user-attachments/assets/673ccf42-7e54-404c-b82f-2d706cc3c64c)*  

### **2. Semantic Search Test**  
| Query | Top Result | Similarity Score |
|-------|------------|------------------|
| "database" | "PostgreSQL vectors" | `0.87` |
| "AI models" | "OpenAI embeddings" | `0.92` |

*![similarscore](https://github.com/user-attachments/assets/de0bc202-7c89-4075-abed-ad07d7b22988)*  

---

## **🚀 Benefits & Use Cases**  
### **Why Use This?**  
🔹 **Better Search** – Finds conceptually similar content  
🔹 **Scalable** – Handles millions of vectors  
🔹 **Cloud-Native** – Azure PostgreSQL ensures high availability  

### **Use Cases**  
- **Document Retrieval** (legal, research)  
- **Chatbot Knowledge Base**  
- **Recommendation Systems**  

---

## **🛠 Troubleshooting**  
| Issue | Fix |
|-------|-----|
| `pgvector` not installed | Run `CREATE EXTENSION vector;` |
| OpenAI 401 Error | Check `.env` API key |
| Slow searches | Add HNSW index:  
  ```sql
  CREATE INDEX ON document_embeddings USING hnsw (embedding vector_cosine_ops);
  ```  

---

## **🎯 Conclusion**  
This system provides a **scalable, AI-powered search** solution using OpenAI and PostgreSQL. By following this guide, you can deploy it in **Azure** or any cloud environment.  

*![deepseek_mermaid_20250507_4b98e2](https://github.com/user-attachments/assets/13d8e39d-eb64-4199-a54d-7d4fab7ea93c)*  

---
**🚀 Next Steps:**  
- Try batch processing 10,000+ documents  
- Integrate with **Azure AI Search** for hybrid search  
- Optimize with **HNSW indexing**  


