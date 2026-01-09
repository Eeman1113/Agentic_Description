import subprocess
import json
import csv
import base64
import ollama
import time

# --- Configuration ---
OLLAMA_MODEL = "qwen3-vl:latest" 
OUTPUT_FILE = "qwen3_agent_descriptions.csv"
REPO_LIMIT = 200  # Increased to catch all your projects

# --- TOOL CLASS ---
class RepoInspector:
    def __init__(self, repo_full_name):
        self.repo_full_name = repo_full_name

    def list_files(self) -> str:
        """
        Lists files in the repository. Call this FIRST to see the project structure.
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
    
    # Map for manual execution
    tool_map = {'list_files': inspector.list_files, 'read_file': inspector.read_file}

    # Strict System Prompt
    system_prompt = """
    You are an automated code analysis agent.
    Your goal is to inspect a GitHub repository and write a ONE-SENTENCE technical description.
    
    RULES:
    1. START by calling `list_files` to see what is inside.
    2. READ key files (README.md, Cargo.toml, package.json) using `read_file`.
    3. DO NOT output JSON plans in your text response. USE THE PROVIDED TOOLS DIRECTLY.
    4. If you have enough info, output the description.
    
    Format: "[Adjective/Tech] [Noun] that [Verb] [Outcome]."
    Example: "A Rust-based grammar engine that optimizes syntax checking using n-gram analysis."
    """

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"Analyze repository: {repo_name}"}
    ]
    
    final_description = "Analysis failed."

    # Loop for Multi-turn (Limit 6 turns)
    for step in range(6): 
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=available_tools,
            stream=True,
            think=True, 
        )

        thinking_buffer = ""
        content_buffer = ""
        tool_calls_buffer = []
        
        in_thinking = False
        
        # --- STREAM PROCESSING ---
        for chunk in response:
            # Handle Thinking
            if chunk.message.thinking:
                if not in_thinking:
                    print("\n🧠 THINKING:", end=" ", flush=True)
                    in_thinking = True
                print(chunk.message.thinking, end="", flush=True)
                thinking_buffer += chunk.message.thinking

            # Handle Content
            if chunk.message.content:
                if in_thinking:
                    print("\n\n💬 RESPONSE:", end=" ", flush=True)
                    in_thinking = False
                print(chunk.message.content, end="", flush=True)
                content_buffer += chunk.message.content

            # Handle Tool Calls
            if chunk.message.tool_calls:
                tool_calls_buffer.extend(chunk.message.tool_calls)
        
        print("\n")

        # Prepare message history
        assistant_msg = {'role': 'assistant', 'content': content_buffer}
        if thinking_buffer: assistant_msg['thinking'] = thinking_buffer
        if tool_calls_buffer: assistant_msg['tool_calls'] = tool_calls_buffer
        
        messages.append(assistant_msg)

        # --- LOGIC CONTROL ---
        
        # 1. If tools were called, execute them
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
            continue # Loop back to let model read the result

        # 2. Heuristic Check: Did the model hallucinate a plan?
        # If content mentions "action": "read_file" but no tool call happened, we nudge it.
        if "action" in content_buffer and "read_file" in content_buffer and not tool_calls_buffer:
            print("⚠️  Model hallucinated a JSON plan. Nudging to use real tools...")
            messages.append({'role': 'user', 'content': "You wrote a plan but didn't trigger the tool. Please properly invoke the 'read_file' or 'list_files' tool function now."})
            continue

        # 3. If no tools and no hallucination, we assume it's the final answer
        if content_buffer.strip():
            final_description = content_buffer
            break
            
    return final_description

# --- MAIN ---

def get_repos():
    print("🔍 Fetching ALL repositories...")
    # Removed date filter to get all repos
    cmd = ['gh', 'search', 'repos', '--owner=@me', '--limit', str(REPO_LIMIT), '--json', 'name,fullName']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error fetching repos. Make sure 'gh auth login' is done.")
        return []
    return json.loads(result.stdout)

def main():
    repos = get_repos()
    print(f"📦 Found {len(repos)} repositories.")

    if not repos:
        return

    # Use 'a' mode (append) in case script crashes, so we don't lose progress
    # But for a clean start, we use 'w'. 
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Repo Name', 'Description'])

    for i, repo in enumerate(repos):
        print(f"\n[{i+1}/{len(repos)}] Processing {repo['name']}...")
        
        try:
            desc = run_agentic_analysis(repo['name'], repo['fullName'])
            
            # Formatting cleanup
            clean_desc = desc.replace('\n', ' ').replace('"', '').strip()
            
            # Save strictly to CSV immediately
            with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([repo['name'], clean_desc])
                
        except Exception as e:
            print(f"❌ Critical Error on {repo['name']}: {e}")
            with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([repo['name'], "Error during processing"])

    print(f"\n✅ Completed. Check {OUTPUT_FILE}")

if __name__ == "__main__":
    main()