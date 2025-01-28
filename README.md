# AI Engineering Roadmap (Crash Course)

### **Level 1 (Beginner)**

1. Understand the basics of LLM - you should just know how ChatGPT works at a high level
    - Article: [What is an LLM?](https://www.datacamp.com/blog/what-is-an-llm-a-guide-on-large-language-models)
    - Video: [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)
2. Learn Prompt Engineering for Developers. How to write prompts to improve the response of an LLM.
    - Articles
        - [Introduction to prompt engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering)
        - [Prompt engineering guide](https://platform.openai.com/docs/guides/prompt-engineering)
    - Video:
        - [https://www.promptingguide.ai/techniques/zeroshot](https://www.promptingguide.ai/techniques/zeroshot)
        - [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
3. Learn to call closed and open-source LLM models, function calling, passing prompts, and parsing responses
    - Articles
        - [Using GPT-3.5 and GPT-4 via the OpenAI API in Python](https://www.datacamp.com/tutorial/using-gpt-models-via-the-openai-api-in-python)
        - [Cracking Open the OpenAI (Python) API](https://towardsdatascience.com/cracking-open-the-openai-python-api-230e4cae7971)
        - [Developer quickstart from OpenAI](https://platform.openai.com/docs/quickstart)
    - Video: [Building Systems with the ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/)
4. Learn to create and automate a sequence of operations - Chains using langchain.
    - Articles
        - [LangChain: Introduction and Getting Started](https://www.pinecone.io/learn/series/langchain/langchain-intro/)
        - [Getting started with LangChain](https://www.pluralsight.com/resources/blog/data/getting-started-langchain)
        - [How to Build LLM Applications with LangChain Tutorial](https://www.datacamp.com/tutorial/how-to-build-llm-applications-with-langchain)
    - Video: [LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
5. Basic app development using Streamlit or Gradio for POCs and demos.
    - Articles
        - [Streamlit tutorial](https://www.datacamp.com/tutorial/streamlit)
        - [Streamlit getting started](https://docs.streamlit.io/get-started/tutorials/create-an-app)
        - [Build an LLM app using LangChain](https://docs.streamlit.io/develop/tutorials/llms/llm-quickstart)
    - Video: [Streamlit: The Fastest Way To Build Python Apps?](https://www.youtube.com/watch?v=D0D4Pa22iG0)

### **Level 2 (Intermediate)**

1. Understanding vector embeddings and vector databases
    - Articles
        - [What are Vector Embeddings?](https://qdrant.tech/articles/what-are-embeddings/)
        - [An Intuitive 101 Guide to Vector Embeddings](https://medium.com/@2twitme/an-intuitive-101-guide-to-vector-embeddings-ffde295c3558)
        - [A Gentle Introduction to Vector Databases](https://weaviate.io/blog/what-is-a-vector-database)
    - Video: [Vector Databases: from Embeddings to Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)
2. Learning how to use vector databases for your application
    - Articles
        - [Mastering Vector Databases with Pinecone Tutorial](https://www.datacamp.com/tutorial/mastering-vector-databases-with-pinecone-tutorial)
        - [Build a Question Answering App Using Pinecone And Python](https://betterprogramming.pub/build-a-question-answering-app-using-pinecone-and-python-1d624c5818bf)
    - Video: [Building Applications with Vector Databases](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/)
3. Building retrieval-augmented Generation (RAG) - chat with your knowledge base
    - Articles
        - [What is Retrieval Augmented Generation (RAG)?](https://www.datacamp.com/blog/what-is-retrieval-augmented-generation-rag)
        - [Build a Retrieval Augmented Generation (RAG) App](https://python.langchain.com/v0.2/docs/tutorials/rag/)
        - [Hands-On with RAG: Step-by-Step Guide to Integrating Retrieval Augmented Generation in LLMs](https://blog.demir.io/hands-on-with-rag-step-by-step-guide-to-integrating-retrieval-augmented-generation-in-llms-ac3cb075ab6f)
    - Video: [LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)

### **Level 3 (Advanced)**

1. Understanding hybrid search
    - Articles
        - [The Basics of AI-Powered (Vector) Search](https://cameronrwolfe.substack.com/p/the-basics-of-ai-powered-vector-search)
        - [On Hybrid Search](https://qdrant.tech/articles/hybrid-search/)
    - Video: [Large Language Models with Semantic Search](https://www.deeplearning.ai/short-courses/large-language-models-semantic-search/)
2. Evaluating RAG
    - Articles
        - [Evaluating Retrieval Augmented Generation](https://superlinked.com/vectorhub/articles/evaluating-retrieval-augmented-generation-framework)
        - [RAG Evaluation: Don’t let customers tell you first](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)
    - Video: [Building and Evaluating Advanced RAG Applications](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)
3. Build multi-modal applications - hybrid semantic search with text and image
    - Articles
        - [What is Multimodal Search](https://cloud.google.com/blog/products/ai-machine-learning/multimodal-generative-ai-search)
        - [Multi-modal Image Search with Embeddings & Vector DBs](https://medium.com/@tenyks_blogger/multi-modal-image-search-with-embeddings-vector-dbs-cee61c70a88a)
    - Video: [Building Multimodal Search and RAG](https://www.deeplearning.ai/short-courses/building-multimodal-search-and-rag/)
4. Building agents - iterative workflows to finish off a big task
    - Articles
        - [What is Agent](https://docs.deepwisdom.ai/main/en/guide/tutorials/concepts.html#agent)
        - [Agentic Design Patterns](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/)
        - [The Future of Generative AI is Agentic: What You Need to Know](https://towardsdatascience.com/the-future-of-generative-ai-is-agentic-what-you-need-to-know-01b7e801fa69)
        - [Agentic RAG](https://www.leewayhertz.com/agentic-rag/#Types-of-agentic-RAG-based-on-function)
    - Videos
        - [What's next for AI agentic workflows ft. Andrew Ng of AI Fund](https://www.youtube.com/watch?v=sal78ACtGTc)
        - [Functions, Tools and Agents with LangChain](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/)
        - [Building Agentic RAG with LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/)
5. Building multi-agent applications where more than one agent works together to provide a better solution
    - Articles
        - [Agentic Design Patterns Part 5, Multi-Agent Collaboration](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/)
        - [Multi-Agent System](https://abvijaykumar.medium.com/multi-agent-architectures-e09c53c7fe0d)
    - Video: [Multi AI Agent Systems with crewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)

### **Level 4 (Expert)**

1. Evaluate and benchmark the model performance
    - Video: [Evaluating and Debugging Generative AI Models Using Weights and Biases](https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/)
2. LLMOps - build complete e2e pipelines with model registry, observability and automated testing
    - Videos
        - [LLMOps](https://www.deeplearning.ai/short-courses/llmops/)
        - [Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)
3. Secure your AI applications using techniques like prompt hacking and incorporating defensive measures by checking against vulnerabilities and potential risks
    - Videos
        - [Quality and Safety for LLM Applications](https://www.deeplearning.ai/short-courses/quality-safety-llm-applications/)
        - [Red Teaming LLM Applications](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/)
4. Fine-tuning pre-trained LLMs for domain-specific knowledge
    - Video: [Finetuning Large Language Models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)
  
### ps: I am making youtube videos to share the similar content. You can subscribe it too. [BlogYourCode](https://www.youtube.com/watch?v=U93RWtA5cCo)
