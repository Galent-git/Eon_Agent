
# eon_agent.py (Main Script)
import json
import time
from datetime import datetime
import random
import os
import logging
import importlib # Used for checking function existence more reliably

# --- LLM Integration ---
# Import necessary libraries only if they will be used
try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False

try:
    import google.generativeai as genai
    google_available = True
except ImportError:
    google_available = False

try:
    import requests
    requests_available = True
except ImportError:
    requests_available = False

logger = logging.getLogger("EonAgent")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")

# --- Agent Class ---
class EonAgent:
    def __init__(self, memory_file="memory.json", dream_file="dreams.md", script_file="eon_agent.py", llm_provider="openai", model_name=None):
        self.memory_file = memory_file
        self.dream_file = dream_file
        self.script_file = script_file # Used conceptually now, not for direct writing/execution
        self.llm_provider = llm_provider.lower()
        self.model_name = model_name
        self.memory = self.load_memory()

        # Configure LLM Client based on provider
        self._configure_llm()

    def _configure_llm(self):
        """Configure the LLM client based on the chosen provider."""
        if self.llm_provider == "openai":
            if not openai_available:
                raise ImportError("OpenAI library not installed. `pip install openai`")
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            self.model_name = self.model_name or "gpt-3.5-turbo" # Default OpenAI model
            logger.info(f"Using OpenAI provider with model: {self.model_name}")

        elif self.llm_provider == "google":
            if not google_available:
                raise ImportError("Google GenerativeAI library not installed. `pip install google-generativeai`")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            genai.configure(api_key=google_api_key)
            self.model_name = self.model_name or "gemini-1.5-flash-latest" # Default Google model
            logger.info(f"Using Google provider with model: {self.model_name}")
            self.llm_client = genai.GenerativeModel(self.model_name)

        elif self.llm_provider == "ollama":
            if not requests_available:
                raise ImportError("Requests library not installed. `pip install requests`")
            self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
            self.model_name = self.model_name or "mistral" # Default Ollama model
            logger.info(f"Using Ollama provider with model: {self.model_name} at {self.ollama_url}")
            # Test connection
            try:
                response = requests.head(os.getenv("OLLAMA_URL", "http://localhost:11434"), timeout=2)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                 logger.warning(f"Could not connect to Ollama server at {self.ollama_url}. Ensure it's running. Error: {e}")
                 # Decide if this should be a fatal error or just a warning
                 # raise ConnectionError(f"Could not connect to Ollama server at {self.ollama_url}") from e


        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Choose 'openai', 'google', or 'ollama'.")

    def load_memory(self):
        """Load the memory from a JSON file."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r") as f:
                    return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error reading memory file {self.memory_file}. Starting with empty memory.")
        return []

    def save_memory(self):
        """Save the current memory to a JSON file."""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.memory, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving memory to {self.memory_file}: {e}")

    def log_dream(self, dream):
        """Append the generated dream to the dream log file with a timestamp."""
        timestamp = datetime.now().isoformat()
        try:
            with open(self.dream_file, "a") as f:
                f.write(f"- {timestamp}: {dream}\n")
        except IOError as e:
            logger.error(f"Error writing to dream file {self.dream_file}: {e}")


    def get_last_dream(self):
        """Return the last recorded dream from the dream log."""
        if os.path.exists(self.dream_file):
            try:
                with open(self.dream_file, "r") as f:
                    dreams = f.readlines()
                if dreams:
                    return dreams[-1].strip().split(": ", 1)[-1] # Get text after timestamp
            except IOError as e:
                 logger.error(f"Error reading dream file {self.dream_file}: {e}")
        return "No dreams logged yet."

    def call_llm(self, prompt, temperature=0.7, max_tokens=150, top_p=1.0):
        """Call the configured LLM provider to generate a response."""
        logger.debug(f"Calling LLM ({self.llm_provider}) with prompt: '{prompt[:100]}...'")
        try:
            if self.llm_provider == "openai":
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                return response.choices[0].message.content.strip()

            elif self.llm_provider == "google":
                 # Adjust generation config as needed for Gemini
                 generation_config = genai.types.GenerationConfig(
                     max_output_tokens=max_tokens,
                     temperature=temperature,
                     top_p=top_p)
                 response = self.llm_client.generate_content(prompt, generation_config=generation_config)
                 return response.text.strip()


            elif self.llm_provider == "ollama":
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": { # Ollama uses 'options'
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_tokens # Corresponds roughly to max_tokens
                    }
                }
                response = requests.post(self.ollama_url, json=payload)
                response.raise_for_status() # Raise an exception for bad status codes
                # The response structure might vary slightly based on Ollama version
                return response.json().get('response', '').strip()

        except Exception as e:
            logger.error(f"Error calling LLM ({self.llm_provider}): {e}")
            return f"[LLM Error: {e}]"

    # --- Conceptual Self-Writing Simulation ---
    def check_or_simulate_function(self, function_name, function_description):
        """
        Checks if a function exists locally. If not, logs that it *would*
        generate it, but does NOT modify the script. Returns True if exists, False otherwise.
        """
        # Attempt to find the function in the current module's scope
        # This assumes functions are defined in this file or imported
        current_module = importlib.import_module(__name__)
        if hasattr(current_module, function_name) and callable(getattr(current_module, function_name)):
             logger.info(f"Function '{function_name}' found locally.")
             return True
        else:
            logger.warning(f"Function '{function_name}' not found locally.")
            # Simulate the self-writing step conceptually
            logger.info(f"[Simulation] Would attempt to generate '{function_name}' via LLM based on description: '{function_description}'")
            # In a real (unsafe) version, LLM call and file append would happen here.
            # We return False to indicate it wasn't found or truly 'created'.
            return False

    def execute_query(self, query):
        """
        Process the query. Simulate self-writing for specific functions,
        otherwise use LLM for a general response.
        """
        result = ""
        if "report" in query.lower():
            function_name = "generate_report"
            function_description = ("takes data string, processes it (e.g., counts words), "
                                    "and returns a formatted string report.")

            # Check if function exists (or simulate the check/generation)
            function_exists = self.check_or_simulate_function(function_name, function_description)

            if function_exists:
                try:
                    # Call the *actual* pre-defined function
                    report_func = getattr(importlib.import_module(__name__), function_name)
                    result = report_func(query) # Pass the query as data for the example
                    result = f"{result}\n[Used local '{function_name}' function]"
                except Exception as e:
                    logger.error(f"Error executing local function '{function_name}': {e}")
                    result = f"[Error executing local function: {e}]"
            else:
                # Function doesn't exist, provide a placeholder response
                result = f"[Function '{function_name}' not available. LLM generation simulated.]"
                # Optionally, you could ask the main LLM to *simulate* the report
                # prompt_simulate = f"Simulate the output of a function that generates a report from this data: '{query}'"
                # result += "\n" + self.call_llm(prompt_simulate, temperature=0.5)

        elif "summarize memory" in query.lower():
             function_name = "summarize_memory_func"
             function_description = ("takes the agent's memory (list of dicts), analyzes it, "
                                     "and returns a concise summary string.")
             function_exists = self.check_or_simulate_function(function_name, function_description)
             if function_exists:
                 try:
                     summary_func = getattr(importlib.import_module(__name__), function_name)
                     result = summary_func(self.memory)
                     result = f"{result}\n[Used local '{function_name}' function]"
                 except Exception as e:
                     logger.error(f"Error executing local function '{function_name}': {e}")
                     result = f"[Error executing local function: {e}]"
             else:
                  result = f"[Function '{function_name}' not available. LLM generation simulated.]"

        else:
            # For other queries, use the LLM directly
            logger.info("Processing general query via LLM.")
            prompt = f"Process the following query in a concise, practical manner: '{query}'."
            result = self.call_llm(prompt, temperature=0.8, max_tokens=100) # Increased tokens slightly

        return result

    def generate_dream(self, last_task):
        """Generate a surreal dream based on the last task."""
        logger.info("Generating dream...")
        prompt = (f"Reviewing the last task: '{last_task}'. Based on this, generate a surreal, "
                  "poetic, and symbolic dream reflection. Use vivid imagery. Be abstract.")
        # Use higher temperature for more creative/random dreams
        dream = self.call_llm(prompt, temperature=1.3, max_tokens=100, top_p=0.9)
        return dream

    def log_query(self, query, result):
        """Log the query and result to memory."""
        entry = {
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self.memory.append(entry)
        # Optional: Limit memory size
        # MAX_MEMORY = 100
        # self.memory = self.memory[-MAX_MEMORY:]
        self.save_memory() # Save after each query

    def run_cycle(self, query):
        """Runs one complete cycle of the agent."""
        logger.info("=== EonAgent Cycle Start ===")

        # Step 1: Report last dream
        last_dream = self.get_last_dream()
        print(f"[Eon] Recalling Last Dream: {last_dream}")

        # Step 2: Process query
        logger.info(f"Processing Query: {query}")
        result = self.execute_query(query)
        print(f"[Eon] Task Output: {result}")

        # Step 3: Log query work
        self.log_query(query, result)
        logger.info("Query and result logged to memory.")

        # Step 4: Generate dream
        dream = self.generate_dream(f"Processed query '{query}' and produced output.")
        print(f"[Eon] Dreaming: {dream}")

        # Step 5: Log dream
        self.log_dream(dream)
        logger.info("Dream logged.")

        logger.info("=== EonAgent Cycle Complete ===\n")

# --- Pre-defined Functions (Examples for the Agent to Find) ---
# These would normally be checked by check_or_simulate_function

def generate_report(data_string: str) -> str:
    """
    A pre-defined function to simulate report generation.
    (Replace with actual logic if needed)
    """
    logger.info(f"Executing pre-defined 'generate_report' function with data: '{data_string[:50]}...'")
    word_count = len(data_string.split())
    # Simulate some processing
    report = f"Report Generated:\n"
    report += f"- Input Data Snippet: {data_string[:50]}...\n"
    report += f"- Analysis: Input contains approximately {word_count} words.\n"
    report += f"- Status: Completed."
    return report

def summarize_memory_func(memory_list: list) -> str:
    """
    A pre-defined function to simulate memory summarization.
    """
    logger.info(f"Executing pre-defined 'summarize_memory_func'...")
    num_entries = len(memory_list)
    if num_entries == 0:
        return "Memory Summary: No entries recorded yet."

    summary = f"Memory Summary: Contains {num_entries} entries.\n"
    # Add more sophisticated summary logic here if desired
    # For example, count types of queries, find common themes, etc.
    last_entry = memory_list[-1]
    summary += f"Most Recent Query: '{last_entry.get('query', 'N/A')[:50]}...'"
    return summary


# --- Main Execution Loop ---
if __name__ == "__main__":
    # Example usage: Choose provider via environment variable or default
    llm_choice = os.getenv("LLM_PROVIDER", "ollama").lower() # Default to Ollama
    model_choice = os.getenv("MODEL_NAME", None) # Let defaults in class handle it if None

    agent = EonAgent(llm_provider=llm_choice, model_name=model_choice)

    # Example Cycle
    agent.run_cycle("Generate a status report on project alpha.")
    time.sleep(2) # Pause for effect/readability
    agent.run_cycle("Summarize memory contents.")
    time.sleep(2)
    agent.run_cycle("What are the key challenges in deploying large language models?")

    # Interactive loop example (optional)
    # print("\nEntering interactive mode. Type 'quit' to exit.")
    # while True:
    #     user_query = input("> ")
    #     if user_query.lower() == 'quit':
    #         break
    #     agent.run_cycle(user_query)
    #     time.sleep(1)
