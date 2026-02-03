import os
import sys
import time

print("\n" + "="*50)
print("🔍 DIAGNOSTIC TOOL STARTING...")
print("="*50)

# 1. Check Python Version
print(f"✅ Python is running: {sys.version.split()[0]}")

# 2. Check Folders
current_dir = os.getcwd()
print(f"📂 Current Folder: {current_dir}")

doc_path = os.path.join(current_dir, "documents")
if os.path.exists(doc_path):
    files = os.listdir(doc_path)
    if files:
        print(f"✅ Found 'documents' folder with {len(files)} files: {files}")
    else:
        print("❌ 'documents' folder exists but is EMPTY.")
else:
    print("❌ 'documents' folder NOT FOUND.")
    print(f"   (I looked here: {doc_path})")

# 3. Check Imports (One by one to catch the crasher)
print("\n🔄 Checking Libraries (this might take a second)...")
try:
    import langchain_community
    print("✅ langchain-community is installed.")
except ImportError as e:
    print(f"❌ ERROR: Missing Library. {e}")
    print("👉 FIX: run 'pip install langchain-community'")

try:
    import faiss
    print("✅ faiss is installed.")
except ImportError as e:
    print(f"❌ ERROR: Missing Library. {e}")
    print("👉 FIX: run 'pip install faiss-cpu'")

try:
    from langchain_ollama import ChatOllama
    print("✅ langchain-ollama is installed.")
except ImportError as e:
    print(f"❌ ERROR: Missing Library. {e}")
    print("👉 FIX: run 'pip install langchain-ollama'")

# 4. Check Ollama Connection
print("\n🦙 Testing Ollama Connection...")
try:
    llm = ChatOllama(model="llama3")
    res = llm.invoke("Hi")
    print("✅ SUCCESS! Connected to Llama 3.")
except Exception as e:
    print("❌ ERROR: Could not connect to Ollama.")
    print(f"   Reason: {e}")
    print("👉 FIX: Ensure 'ollama serve' is running or background process is active.")

print("\n" + "="*50)
print("🏁 DIAGNOSTIC COMPLETE")
print("="*50)

# 5. PREVENT CLOSING
input("\nPress Enter to exit...")