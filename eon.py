import json
import time
from datetime import datetime
import random
import os
import logging
import importlib # Used for checking function existence more reliably
import sys # For sys.exit

# Load environment variables from .env file FIRST
from dotenv import load_dotenv
load_dotenv()

# --- LLM Integration ---
# (Keep the try-except imports as they are)
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
    # requests is needed for Ollama, which might be the default
    # Consider making it non-optional or handling this more gracefully
    requests_available = False


# --- Logging Setup ---
logger = logging.getLogger("EonAgent")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")

# --- Agent Class ---
class EonAgent:
    def __init__(self,
                 memory_file=None, # Default handled below
                 dream_file=None,  # Default handled below
                 script_file="eon_agent.py", # Keep for concept reference
                 llm_provider=None,# Default handled below
                 model_name=None): # Default handled within _configure_llm

        # Configuration from environment variables or defaults
        self.memory_file = memory_file or os.getenv("MEMORY_FILE", "memory.json")
        self.dream_file = dream_file or os.getenv("DREAM_FILE", "dreams.md")
        self.script_file = script_file # Primarily conceptual reference now
        self.llm_provider = (llm_provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        self.model_name = model_name or os.getenv("MODEL_NAME") # Specific model default handled per provider

        logger.info(f"Initializing EonAgent: LLM Provider='{self.llm_provider}', Memory='{self.memory_file}', Dreams='{self.dream_file}'")

        self.memory = self.load_memory()

        # Configure LLM Client based on provider
        try:
            self._configure_llm()
        except (ImportError, ValueError, ConnectionError) as e:
            logger.error(f"Fatal Error during LLM configuration: {e}")
            sys.exit(1) # Exit if core component fails setup


    def _configure_llm(self):
        """Configure the LLM client based on the chosen provider."""
        if self.llm_provider == "openai":
            if not openai_available:
                raise ImportError("OpenAI library not installed. Run: pip install openai python-dotenv")
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key or openai.api_key == "your-openai-api-key-here":
                raise ValueError("OPENAI_API_KEY environment variable not set or is placeholder.")
            # If model_name wasn't passed to __init__ or set via env, use default
            self.model_name = self.model_name or "gpt-3.5-turbo"
            logger.info(f"Using OpenAI provider with model: {self.model_name}")
            # OpenAI client is used via static methods like openai.chat.completions.create

        elif self.llm_provider == "google":
            if not google_available:
                raise ImportError("Google GenerativeAI library not installed. Run: pip install google-generativeai python-dotenv")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key or google_api_key == "your-google-api-key-here":
                raise ValueError("GOOGLE_API_KEY environment variable not set or is placeholder.")
            genai.configure(api_key=google_api_key)
             # If model_name wasn't passed to __init__ or set via env, use default
            self.model_name = self.model_name or "gemini-1.5-flash-latest"
            logger.info(f"Using Google provider with model: {self.model_name}")
            self.llm_client = genai.GenerativeModel(self.model_name) # Instantiate the client

        elif self.llm_provider == "ollama":
            if not requests_available:
                # If Ollama is chosen, requests becomes mandatory
                raise ImportError("Requests library not installed. Run: pip install requests python-dotenv")
            self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
            self.ollama_check_url = os.getenv("OLLAMA_CHECK_URL", "http://localhost:11434")
             # If model_name wasn't passed to __init__ or set via env, use default
            self.model_name = self.model_name or "mistral" # Or llama3? Set a consistent default.
            logger.info(f"Using Ollama provider with model: {self.model_name} at {self.ollama_url}")
            # Test connection
            try:
                # Use HEAD request for efficiency, or GET to a known endpoint
                response = requests.head(self.ollama_check_url, timeout=3)
                # Some servers might not support HEAD on base URL, try GET as fallback
                if not response.ok:
                    response = requests.get(self.ollama_check_url, timeout=3)
                response.raise_for_status()
                logger.info(f"Successfully connected to Ollama server at {self.ollama_check_url}")
            except requests.exceptions.RequestException as e:
                 logger.error(f"Could not connect to Ollama server at {self.ollama_check_url}. Ensure it's running.")
                 # Make this fatal if Ollama is the chosen provider
                 raise ConnectionError(f"Ollama connection failed at {self.ollama_check_url}") from e

        else:
            raise ValueError(f"Unsupported LLM provider: '{self.llm_provider}'. Choose 'openai', 'google', or 'ollama'. Check LLM_PROVIDER in .env file.")

    # (Keep load_memory, save_memory, log_dream, get_last_dream as they are)
    def load_memory(self):
        """Load the memory from a JSON file."""
        # Add check if file exists before trying to open
        if not os.path.exists(self.memory_file):
            logger.info(f"Memory file '{self.memory_file}' not found. Starting with empty memory.")
            return []
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f: # Add encoding
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from memory file {self.memory_file}. Starting with empty memory.")
            return []
        except Exception as e:
             logger.error(f"Error reading memory file {self.memory_file}: {e}")
             return [] # Return empty list on other read errors too

    def save_memory(self):
        """Save the current memory to a JSON file."""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f: # Add encoding
                json.dump(self.memory, f, indent=2, ensure_ascii=False) # Add ensure_ascii=False
        except IOError as e:
            logger.error(f"Error saving memory to {self.memory_file}: {e}")

    def log_dream(self, dream):
        """Append the generated dream to the dream log file with a timestamp."""
        timestamp = datetime.now().isoformat()
        try:
            with open(self.dream_file, "a", encoding="utf-8") as f: # Add encoding
                f.write(f"--- Dream @ {timestamp} ---\n")
                f.write(f"{dream}\n\n") # Add separation
        except IOError as e:
            logger.error(f"Error writing to dream file {self.dream_file}: {e}")

    def get_last_dream(self):
        """Return the last recorded dream from the dream log."""
        if not os.path.exists(self.dream_file):
            return "No dreams logged yet."
        try:
            with open(self.dream_file, "r", encoding="utf-8") as f: # Add encoding
                content = f.read().strip()
            if not content:
                return "Dream log is empty."
            # Find the last dream block
            last_dream_marker = "--- Dream @"
            last_marker_pos = content.rfind(last_dream_marker)
            if last_marker_pos == -1:
                return "Could not parse last dream from log."
            # Get content after the last marker's line
            dream_text = content[last_marker_pos:].split("\n", 1)[-1].strip()
            return dream_text
        except IOError as e:
                logger.error(f"Error reading dream file {self.dream_file}: {e}")
                return "[Error reading dream log]"
        except Exception as e: # Catch other potential errors during parsing
            logger.error(f"Error parsing dream file {self.dream_file}: {e}")
            return "[Error parsing dream log]"


    def call_llm(self, prompt, temperature=0.7, max_tokens=200, top_p=1.0): # Increased default max_tokens
        """Call the configured LLM provider to generate a response."""
        logger.debug(f"Calling LLM ({self.llm_provider}, model: {self.model_name}) with prompt: '{prompt[:100]}...'")
        try:
            if self.llm_provider == "openai":
                # Ensure API key is available (might have been cleared)
                if not openai.api_key: raise ValueError("OpenAI API key not configured.")
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                # Add basic check for response structure
                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content.strip()
                else:
                    logger.error("Invalid response structure received from OpenAI.")
                    return "[LLM Error: Invalid response structure]"


            elif self.llm_provider == "google":
                # Ensure client exists
                if not hasattr(self, 'llm_client'): raise ValueError("Google GenAI client not initialized.")
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p)
                # Handle potential safety blocks or empty responses
                response = self.llm_client.generate_content(prompt, generation_config=generation_config)
                if response.parts:
                     return response.text.strip()
                else:
                     # Log safety ratings if available
                     if response.prompt_feedback:
                         logger.warning(f"Google GenAI response blocked or empty. Feedback: {response.prompt_feedback}")
                     else:
                         logger.warning("Google GenAI response blocked or empty.")
                     return "[LLM Warning: Response blocked or empty]"


            elif self.llm_provider == "ollama":
                 payload = {
                     "model": self.model_name,
                     "prompt": prompt,
                     "stream": False,
                     "options": {
                         "temperature": temperature,
                         "top_p": top_p,
                         "num_predict": max_tokens # Max tokens in Ollama
                     }
                 }
                 response = requests.post(self.ollama_url, json=payload, timeout=60) # Add timeout
                 response.raise_for_status()
                 response_data = response.json()
                 # Check for 'response' key, handle potential variations
                 if 'response' in response_data:
                     return response_data['response'].strip()
                 else:
                     logger.error(f"Unexpected response structure from Ollama: {response_data}")
                     return "[LLM Error: Unexpected response structure]"


        except requests.exceptions.RequestException as e:
             logger.error(f"Network error calling Ollama LLM: {e}")
             return f"[LLM Network Error: {e}]"
        except Exception as e:
            # Catch specific API errors if possible (e.g., openai.APIError)
            logger.error(f"Error calling LLM ({self.llm_provider}): {e}", exc_info=True) # Add exc_info for traceback
            return f"[LLM Error: {type(e).__name__}]"


    # (Keep check_or_simulate_function and _unsafe_generate_and_append_function as they are)
    def check_or_simulate_function(self, function_name, function_description):
        """
        Checks if a function exists locally. If not, logs that it *would*
        generate it, but does NOT modify the script. Returns True if exists, False otherwise.
        """
        # Attempt to find the function in the current module's scope
        # This assumes functions are defined in this file or imported
        try:
            current_module = importlib.import_module(__name__)
            if hasattr(current_module, function_name) and callable(getattr(current_module, function_name)):
                logger.info(f"Function '{function_name}' found locally.")
                return True
            else:
                logger.warning(f"Function '{function_name}' not found locally.")
                # Simulate the self-writing step conceptually
                logger.info(f"[Simulation] Agent would attempt to generate '{function_name}' via LLM based on description: '{function_description}'")
                # In a real (unsafe) version, LLM call and file append would happen here.
                # We return False to indicate it wasn't found or truly 'created'.
                return False
        except Exception as e:
            logger.error(f"Error checking for function '{function_name}': {e}")
            return False # Assume not found if check fails

    def _unsafe_generate_and_append_function(self, function_name, function_description):
         # (Keep the unsafe function exactly as is, with all warnings)
        """
        *** WARNING: EXTREMELY UNSAFE - FOR DEMONSTRATION ONLY ***
        This function demonstrates the original concept of using an LLM
        to generate Python code and append it to the agent's own script file.
        EXECUTING CODE GENERATED BY AN LLM WITHOUT RIGOROUS SANDBOXING,
        VALIDATION, AND SECURITY REVIEW IS HIGHLY DANGEROUS.
        It can lead to arbitrary code execution, security vulnerabilities,
        data loss, and system instability.

        DO NOT CALL THIS FUNCTION IN PRODUCTION OR UNTRUSTED ENVIRONMENTS.
        It is included purely as an illustration of the technical concept explored
        and is NOT used by the agent's default execution cycle.
        Modifying the agent to call this function directly is done at your own
        significant risk.
        *******************************************************************

        Checks if function exists. If not, calls LLM, and appends result to self.script_file.
        Note: Even if code is appended, it won't be usable in the current runtime
              without complex reloading mechanisms.
        """
        logger.warning("Executing _unsafe_generate_and_append_function - FOR DEMO PURPOSES ONLY.")
        function_found_in_file = False
        if os.path.exists(self.script_file):
            try:
                with open(self.script_file, "r", encoding="utf-8") as f:
                    script_content = f.read()
                    # Basic check if function definition seems to exist
                    if f"def {function_name}(" in script_content:
                        function_found_in_file = True
            except IOError as e:
                logger.error(f"[Unsafe Demo] Error reading script file {self.script_file}: {e}")
                script_content = "" # Assume empty if unreadable
        else:
            script_content = ""
            logger.warning(f"[Unsafe Demo] Script file '{self.script_file}' not found.")


        if not function_found_in_file:
            logger.info(f"[Unsafe Demo] Function '{function_name}' signature not found in script file. Generating code via LLM...")
            prompt = (f"Write a complete Python function called '{function_name}' that {function_description}.\n"
                      "Include necessary imports if any within the function scope or assume standard libraries like 'os', 'json', 'datetime' are imported at the top of the file.\n"
                      "The function should be self-contained.\n"
                      "Return ONLY the raw Python code for the function definition block, starting with 'def'. Do not include explanations, ```python markers, or any surrounding text.")

            # Use a potentially different LLM call if generation needs specific parameters
            # Maybe use lower temp for more predictable code generation
            function_code = self.call_llm(prompt, temperature=0.4, max_tokens=400) # More tokens for code, lower temp?

            if isinstance(function_code, str) and function_code.strip().startswith("def "):
                logger.info(f"[Unsafe Demo] LLM generated code snippet starting with: {function_code[:150]}...")
                try:
                    # Append the generated function code to the script file
                    with open(self.script_file, "a", encoding="utf-8") as f:
                        f.write("\n\n# --- Auto-generated by EonAgent (UNSAFE DEMO) ---\n")
                        f.write(function_code.strip()) # Ensure no leading/trailing whitespace from LLM
                        f.write("\n# --- End Auto-generated ---\n")
                    logger.info(f"[Unsafe Demo] Function '{function_name}' code *appended* to {self.script_file}.")
                    logger.warning("[Unsafe Demo] Appended code is NOT dynamically loaded or executed in the current run.")
                except IOError as e:
                    logger.error(f"[Unsafe Demo] Error writing generated code to {self.script_file}: {e}")
                except Exception as e: # Catch other potential errors
                     logger.error(f"[Unsafe Demo] Unexpected error writing generated code: {e}")
            elif isinstance(function_code, str):
                 logger.error(f"[Unsafe Demo] LLM did not return valid-looking function code for '{function_name}'. Received: {function_code[:200]}...")
            else:
                 logger.error(f"[Unsafe Demo] LLM call failed or returned non-string for '{function_name}'.")
        else:
            logger.info(f"[Unsafe Demo] Function '{function_name}' signature already present in {self.script_file}.")


    def execute_query(self, query):
        """
        Process the query. Use pre-defined functions if available (via safe check),
        otherwise use LLM for a general response.
        """
        result = ""
        # Normalize query for easier checking
        query_lower = query.lower()

        # --- Function Check Logic ---
        # Define functions the agent might try to use
        potential_functions = {
            "report": ("generate_report", "takes data string, processes it (e.g., counts words), and returns a formatted string report."),
            "summarize memory": ("summarize_memory_func", "takes the agent's memory (list of dicts), analyzes it, and returns a concise summary string.")
            # Add more function triggers and descriptions here
        }

        triggered_function = None
        for keyword, (func_name, func_desc) in potential_functions.items():
            if keyword in query_lower:
                triggered_function = (func_name, func_desc)
                logger.info(f"Query keyword '{keyword}' triggered check for function '{func_name}'.")
                break # Use the first match

        if triggered_function:
            function_name, function_description = triggered_function
            # Check if function exists (this is the SAFE simulation path)
            function_exists = self.check_or_simulate_function(function_name, function_description)

            if function_exists:
                try:
                    # Get the actual pre-defined function from the current module
                    # Ensure the module is fresh in case it was somehow modified (though we don't support that safely)
                    current_module = importlib.reload(importlib.import_module(__name__)) # Reload might be overkill but safer conceptually
                    report_func = getattr(current_module, function_name)

                    # Determine what data to pass based on the function
                    if function_name == "generate_report":
                        func_input = query # Pass the original query as data for this example
                    elif function_name == "summarize_memory_func":
                         func_input = self.memory
                    else:
                         func_input = None # Or handle other functions

                    # Call the function
                    if func_input is not None:
                        result = report_func(func_input)
                    else:
                         result = report_func() # Call without arguments if appropriate

                    result = f"{result}\n[Executed local function: '{function_name}']"

                except AttributeError:
                     logger.error(f"Local function '{function_name}' found by check but failed getattr.")
                     result = f"[Error: Could not access local function '{function_name}' after check.]"
                except Exception as e:
                    logger.error(f"Error executing local function '{function_name}': {e}", exc_info=True)
                    result = f"[Error executing local function '{function_name}': {e}]"
            else:
                # Function doesn't exist, provide a placeholder response indicating simulation
                result = f"[Function '{function_name}' is not available. Agent simulated attempting generation.]"
                # Optionally, use LLM to simulate the function's *output*
                # prompt_simulate = f"Briefly simulate the likely output of a function that would: {function_description}. Base the simulation on the query: '{query}'"
                # simulation_output = self.call_llm(prompt_simulate, temperature=0.5, max_tokens=150)
                # result += f"\n[LLM Output Simulation]:\n{simulation_output}"

        else:
            # No specific function keyword matched, use LLM for a general response
            logger.info("Processing general query via LLM.")
            # Make prompt clearer about agent's role
            prompt = f"As the Eon Agent, process the following user query concisely: '{query}'"
            result = self.call_llm(prompt, temperature=0.7, max_tokens=250) # Adjusted defaults

        return result

    def generate_dream(self, last_task_description):
        """Generate a surreal dream based on the last task."""
        logger.info("Generating dream based on: " + last_task_description)
        prompt = (f"You are the Eon Agent, reflecting on your recent activity.\n"
                  f"Last Activity: {last_task_description}\n"
                  f"Now, generate a surreal, highly symbolic, and poetic dream monologue (1-3 sentences) related to this activity. "
                  f"Use metaphors and abstract imagery. Focus on themes of processing, memory, or code.")
        # Use higher temperature for more creative/random dreams
        dream = self.call_llm(prompt, temperature=1.2, max_tokens=100, top_p=0.9) # Adjusted temp/tokens
        # Basic check if LLM returned an error string
        if dream.startswith("[LLM"):
            logger.warning(f"LLM failed to generate dream, using fallback. Error: {dream}")
            return "Static flickered behind the eyelids; circuits hummed a forgotten tune."
        return dream


    def log_query(self, query, result):
        """Log the query and result to memory."""
        entry = {
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        # Prevent excessively large memory files if result was huge (e.g., error traceback)
        MAX_RESULT_LEN = 1000
        if len(result) > MAX_RESULT_LEN:
             entry["result"] = result[:MAX_RESULT_LEN] + "... [truncated]"

        self.memory.append(entry)
        # Optional: Limit memory size (e.g., keep last 100 entries)
        MAX_MEMORY_ENTRIES = int(os.getenv("MAX_MEMORY_ENTRIES", 100))
        if len(self.memory) > MAX_MEMORY_ENTRIES:
            self.memory = self.memory[-MAX_MEMORY_ENTRIES:]

        self.save_memory() # Save after each query


    def run_cycle(self, query):
        """Runs one complete cycle of the agent."""
        print("\n" + "="*10 + " Eon Agent Cycle Start " + "="*10)

        # Step 1: Report last dream
        last_dream = self.get_last_dream()
        print(f"\n[Eon] Recalling Last Dream:\n{last_dream}\n")
        time.sleep(0.5) # Small pause

        # Step 2: Process query
        print(f"[Eon] Processing Query: '{query}'")
        result = self.execute_query(query)
        print(f"\n[Eon] Task Output:\n{result}\n")
        time.sleep(0.5)

        # Step 3: Log query work
        self.log_query(query, result)
        logger.info("Query and result logged to memory.")

        # Step 4: Generate dream based on the query processing
        dream = self.generate_dream(f"Processed query '{query}' resulting in output starting with '{result[:50]}...'")
        print(f"[Eon] Dreaming...\n{dream}\n")

        # Step 5: Log dream
        self.log_dream(dream)
        logger.info("Dream logged.")

        print("="*10 + " Eon Agent Cycle End " + "="*12 + "\n")


# === Pre-defined Functions (Examples for the Agent to Find) ===
# These MUST exist for the check_or_simulate_function to find them when SAFE simulation runs.

def generate_report(data_string: str) -> str:
    """
    A pre-defined function the agent can 'find' and execute safely.
    Simulates report generation based on input string.
    """
    logger.info(f"Executing pre-defined 'generate_report' function.")
    try:
        word_count = len(data_string.split())
        char_count = len(data_string)
        # Simulate some analysis
        report = f"Analysis Report:\n"
        report += f"- Input Length: {char_count} characters, approximately {word_count} words.\n"
        # Add more complex analysis here if needed
        report += f"- Status: Analysis complete."
        return report
    except Exception as e:
        logger.error(f"Error in generate_report: {e}")
        return "[Error during report generation]"

def summarize_memory_func(memory_list: list) -> str:
    """
    A pre-defined function the agent can 'find' and execute safely.
    Simulates summarizing the agent's memory list.
    """
    logger.info(f"Executing pre-defined 'summarize_memory_func'...")
    try:
        num_entries = len(memory_list)
        if num_entries == 0:
            return "Memory Summary: No entries recorded yet."

        summary = f"Memory Summary: Contains {num_entries} entries.\n"

        # Example: Count how many results were errors
        error_count = sum(1 for entry in memory_list if "[Error" in entry.get("result", ""))
        summary += f"- Entries with errors: {error_count}\n"

        # Example: Get timestamp of the first and last entry
        if num_entries > 0:
            first_ts = memory_list[0].get("timestamp", "N/A")
            last_ts = memory_list[-1].get("timestamp", "N/A")
            summary += f"- Time Range: From {first_ts} to {last_ts}"

        return summary
    except Exception as e:
        logger.error(f"Error in summarize_memory_func: {e}")
        return "[Error during memory summarization]"


# --- Main Execution Loop ---
if __name__ == "__main__":
    try:
        # No need to get provider/model here, __init__ handles it from .env
        agent = EonAgent()

        # Example Cycles
        agent.run_cycle("Please generate a status report based on this input: Project Phoenix is ongoing.")
        time.sleep(1)
        agent.run_cycle("Summarize memory contents.")
        time.sleep(1)
        agent.run_cycle("Explain the concept of emergent behavior in complex systems.")
        time.sleep(1)
        agent.run_cycle("Generate a report detailing the current system load.") # Will trigger function check

        # Optional: Demonstrate the UNSAFE function (USE WITH EXTREME CAUTION)
        # print("\n" + "*"*20 + " WARNING: DEMONSTRATING UNSAFE FUNCTION " + "*"*20)
        # print("This will attempt to use the LLM to write code and APPEND it to eon_agent.py")
        # print("This is purely illustrative and DANGEROUS. DO NOT USE IN PRODUCTION.")
        # confirm = input("Type 'YES_I_UNDERSTAND_THE_RISKS' to proceed: ")
        # if confirm == "YES_I_UNDERSTAND_THE_RISKS":
        #     agent._unsafe_generate_and_append_function(
        #         "calculate_complexity",
        #         "takes a string, calculates a basic complexity score (e.g., length + unique words), and returns the score as an integer."
        #     )
        # else:
        #     print("Skipping unsafe function demonstration.")
        # print("*"*60 + "\n")


        # Optional Interactive loop
        print("\n--- Entering interactive mode (type 'quit' to exit) ---")
        while True:
            try:
                user_query = input("Ask Eon Agent: ")
                if user_query.lower() == 'quit':
                    print("Exiting.")
                    break
                if not user_query: # Handle empty input
                    continue
                agent.run_cycle(user_query)
            except EOFError: # Handle Ctrl+D
                 print("\nExiting.")
                 break
            except KeyboardInterrupt: # Handle Ctrl+C
                 print("\nExiting.")
                 break

    except Exception as main_error:
         logger.critical(f"An unexpected error occurred in the main execution block: {main_error}", exc_info=True)
         print(f"\nA critical error occurred. Please check the logs. Error: {main_error}")
         sys.exit(1)
