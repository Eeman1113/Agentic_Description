import subprocess
import json
import csv
import base64
import ollama
import time
import os

# --- Configuration ---
OLLAMA_MODEL = "qwen3-vl:latest" 
OUTPUT_FILE = "qwen3_agent_descriptions.csv"
REPO_LIMIT = 300

# --- TOOL CLASS ---
class RepoInspector:
    def __init__(self, repo_full_name):
        self.repo_full_name = repo_full_name

    def list_files(self) -> str:
        """
        Lists files in the repository. 
        """
        ignore_paths = ['node_modules', '.git', 'assets', 'dist', 'build', 'vendor', 
                        'public', 'static', 'yarn.lock', 'package-lock.json', '.env', 
                        'images', 'fonts', 'test', 'tests']
        
        # Try HEAD first
        cmd = ['gh', 'api', f'repos/{self.repo_full_name}/git/trees/HEAD?recursive=1', '--jq', '.tree[].path']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Fallback to master
        if result.returncode != 0:
            cmd[2] = f'repos/{self.repo_full_name}/git/trees/master?recursive=1'
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0 or not result.stdout:
            return "Error: Could not list files. Repo might be empty."

        lines = result.stdout.splitlines()
        # Filter noise
        filtered = [l for l in lines if not any(x in l for x in ignore_paths)]
        
        # Prioritize key files
        priority_files = []
        other_files = []
        for f in filtered:
            if any(k in f.lower() for k in ['readme', 'cargo.toml', 'package.json', 'requirements.txt', 'main.py', 'src/', 'index', 'app']):
                priority_files.append(f)
            else:
                other_files.append(f)
        
        final_list = priority_files + other_files
        return "\n".join(final_list[:80]) # Return top 80 most relevant files

    def read_file(self, file_path: str) -> str:
        """
        Reads the content of a specific file.
        Args:
            file_path: The path of the file to read (e.g., 'README.md', 'Cargo.toml').
        """
        print(f"   📖 Reading: {file_path}")
        cmd = ['gh', 'api', f'repos/{self.repo_full_name}/contents/{file_path}', '--jq', '.content']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return f"Error: Could not read {file_path}. File not found."

        try:
            if not result.stdout.strip():
                return "Error: File is empty."
            content = base64.b64decode(result.stdout).decode('utf-8', errors='ignore')
            return content[:6000] # Limit context
        except:
            return "Error: Failed to decode file content."

# --- AGENT LOGIC ---

def run_agentic_analysis(repo_name, repo_full_name):
    print(f"\n⚡ Agent activated for: {repo_name}")
    
    inspector = RepoInspector(repo_full_name)
    available_tools = [inspector.list_files, inspector.read_file]
    tool_map = {'list_files': inspector.list_files, 'read_file': inspector.read_file}

    # 1. PRE-FETCH FILE LIST
    print("   📂 Pre-fetching file list...")
    file_structure = inspector.list_files()

    # 2. UPDATED SYSTEM PROMPT
    system_prompt = """
    You are an automated code analysis agent.
    Your goal is to inspect a GitHub repository and write a ONE-SENTENCE technical description.
    
    RULES:
    1. The file list is provided in the first message. Analyze it to understand the structure.
    2. IMMEDIATELY call `read_file` on key files (e.g., README.md, Cargo.toml, package.json, main.py).
    3. DO NOT ask to list files again.
    4. DO NOT output JSON plans. USE THE TOOLS DIRECTLY.
    5. If you have enough info, output the description.
    
    Format: "[Adjective/Tech] [Noun] that [Verb] [Outcome]."
    Example: "A Rust-based grammar engine that optimizes syntax checking using n-gram analysis."
    """

    # 3. INJECT INTO INITIAL MESSAGE
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"Analyze repository: {repo_name}\n\nHere is the file list:\n{file_structure}"}
    ]
    
    final_description = "Analysis failed."

    # Loop for Multi-turn (Limit 6 turns)
    for step in range(6): 
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=available_tools,
            stream=True,
            # think=True, 
        )

        thinking_buffer = ""
        content_buffer = ""
        tool_calls_buffer = []
        
        in_thinking = False
        
        # --- STREAM PROCESSING ---
        for chunk in response:
            if hasattr(chunk.message, 'thinking') and chunk.message.thinking:
                if not in_thinking:
                    print("\n🧠 THINKING:", end=" ", flush=True)
                    in_thinking = True
                print(chunk.message.thinking, end="", flush=True)
                thinking_buffer += chunk.message.thinking

            if chunk.message.content:
                if in_thinking:
                    print("\n\n💬 RESPONSE:", end=" ", flush=True)
                    in_thinking = False
                print(chunk.message.content, end="", flush=True)
                content_buffer += chunk.message.content

            if chunk.message.tool_calls:
                tool_calls_buffer.extend(chunk.message.tool_calls)
        
        print("\n")

        assistant_msg = {'role': 'assistant', 'content': content_buffer}
        if thinking_buffer: assistant_msg['thinking'] = thinking_buffer
        if tool_calls_buffer: assistant_msg['tool_calls'] = tool_calls_buffer
        
        messages.append(assistant_msg)

        # --- LOGIC CONTROL ---
        if tool_calls_buffer:
            for tool in tool_calls_buffer:
                fname = tool.function.name
                fargs = tool.function.arguments
                
                if fname in tool_map:
                    print(f"🛠️  EXECUTING: {fname} {fargs}...")
                    try:
                        result = tool_map[fname](**fargs)
                    except Exception as e:
                        result = f"Error executing tool: {e}"
                    
                    messages.append({'role': 'tool', 'tool_name': fname, 'content': str(result)})
                else:
                    messages.append({'role': 'tool', 'tool_name': fname, 'content': "Error: Function not found."})
            continue 

        if "action" in content_buffer and "read_file" in content_buffer and not tool_calls_buffer:
            print("⚠️  Model hallucinated a JSON plan. Nudging to use real tools...")
            messages.append({'role': 'user', 'content': "You wrote a plan but didn't trigger the tool. Please properly invoke the 'read_file' tool function now."})
            continue

        if content_buffer.strip():
            final_description = content_buffer
            break
            
    return final_description

# --- MAIN ---

def get_repos():
    print("🔍 Fetching ALL repositories...")
    cmd = ['gh', 'search', 'repos', '--owner=@me', '--limit', str(REPO_LIMIT), '--json', 'name,fullName']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error fetching repos. Make sure 'gh auth login' is done.")
        return []
    return json.loads(result.stdout)

def get_existing_progress(filepath):
    processed = set()
    if not os.path.exists(filepath):
        return processed
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    processed.add(row[0])
    except Exception as e:
        print(f"⚠️ Warning: Could not read existing file: {e}")
    return processed

def main():
    repos = get_repos()
    print(f"📦 Found {len(repos)} repositories total.")

    if not repos:
        return

    processed_repos = get_existing_progress(OUTPUT_FILE)
    print(f"⏭️  Found {len(processed_repos)} repositories already processed.")

    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Repo Name', 'Description'])
            print(f"📄 Created new file: {OUTPUT_FILE}")

    for i, repo in enumerate(repos):
        repo_name = repo['name']
        
        if repo_name in processed_repos:
            print(f"[{i+1}/{len(repos)}] ⏩ Skipping {repo_name} (Already in CSV)")
            continue

        print(f"\n[{i+1}/{len(repos)}] Processing {repo_name}...")
        
        try:
            desc = run_agentic_analysis(repo_name, repo['fullName'])
            clean_desc = desc.replace('\n', ' ').replace('"', '').strip()
            
            with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([repo_name, clean_desc])
                
        except Exception as e:
            print(f"❌ Critical Error on {repo_name}: {e}")
            with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([repo_name, "Error during processing"])

    print(f"\n✅ Completed. Check {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
