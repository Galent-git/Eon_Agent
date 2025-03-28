import json
import time
from datetime import datetime
import random
import os
import openai  # Ensure you have openai installed and configured

class EonAgent:
    def __init__(self, memory_file="memory.json", dream_file="dreams.md", script_file="eon_agent.py"):
        self.memory_file = memory_file
        self.dream_file = dream_file
        self.script_file = script_file
        self.load_memory()
        # Set your OpenAI API key (or configure your local LLM accordingly)
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def load_memory(self):
        """Load the memory from a JSON file."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                self.memory = json.load(f)
        else:
            self.memory = []

    def save_memory(self):
        """Save the current memory to a JSON file."""
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)

    def log_dream(self, dream):
        """Append the generated dream to the dream log file with a timestamp."""
        timestamp = datetime.now().isoformat()
        with open(self.dream_file, "a") as f:
            f.write(f"- {timestamp}: {dream}\n")

    def get_last_dream(self):
        """Return the last recorded dream from the dream log."""
        if os.path.exists(self.dream_file):
            with open(self.dream_file, "r") as f:
                dreams = f.readlines()
            if dreams:
                return dreams[-1].strip()
        return "No dreams logged yet."

    def call_llm(self, prompt, temperature=1.3, max_tokens=60, top_p=0.95):
        """
        Call the LLM (e.g., OpenAI's API) to generate a response.
        Adjust parameters (temperature, max_tokens, top_p) as needed.
        """
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip()

    def check_and_append_function(self, function_name, function_description):
        """
        Inspect the script file to determine if a function is defined.
        If not, use the LLM to generate the code and append it to the script.
        """
        if os.path.exists(self.script_file):
            with open(self.script_file, "r") as f:
                script_content = f.read()
        else:
            script_content = ""
        if f"def {function_name}(" not in script_content:
            print(f"[Eon] Function '{function_name}' not found. Generating code via LLM...")
            prompt = (f"Write a Python function called '{function_name}' that {function_description}.\n"
                      "Return only the code without explanation.")
            function_code = self.call_llm(prompt, temperature=1.0, max_tokens=150, top_p=0.95)
            # Append the generated function code to the script file
            with open(self.script_file, "a") as f:
                f.write("\n\n# Auto-generated function by EonAgent\n" + function_code)
            print(f"[Eon] Function '{function_name}' has been added to the script.")
        else:
            print(f"[Eon] Function '{function_name}' already exists.")

    def execute_query(self, query):
        """
        Process the query. For example, if the query includes 'report', ensure
        that the 'generate_report' function exists (using the self-writing mechanism),
        then execute it. For other queries, ask the LLM to process it.
        """
        result = ""
        if "report" in query.lower():
            # Description of the report-generation function
            function_description = ("generates a report from given data, prints the data, "
                                    "and returns a formatted string report")
            self.check_and_append_function("generate_report", function_description)
            # Dynamically obtain the function (assumes it's now in globals)
            report_func = globals().get("generate_report")
            if callable(report_func):
                try:
                    result = report_func(query)
                    result = f"{result}\n[Used generate_report function]"
                except Exception as e:
                    result = f"Error during execution: {e}"
            else:
                result = "Error: 'generate_report' function is not callable."
        else:
            # For non-report queries, use the LLM to generate a concise practical response
            prompt = f"Process the following query in a concise, practical manner: '{query}'."
            result = self.call_llm(prompt, temperature=0.8, max_tokens=50, top_p=0.9)
        return result

    def generate_dream(self, last_task):
        """
        Generate a high-temperature, surreal dream based on the last task,
        using the LLM.
        """
        prompt = (f"Eon recalls the task: '{last_task}'. Now, in a state of high creativity, "
                  "dream a surreal, poetic reflection on this task. Let your dream be vivid, symbolic, "
                  "and wild.")
        dream = self.call_llm(prompt, temperature=1.3, max_tokens=80, top_p=0.95)
        return dream

    def log_query(self, query, result):
        """Log the processed query along with its result and a timestamp into memory."""
        entry = {
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self.memory.append(entry)
        self.save_memory()

    def run_cycle(self, query):
        """
        Run one complete cycle:
        1. Report the last recorded dream.
        2. Process the incoming query by checking for missing functions, appending code if necessary,
           and executing the query.
        3. Log the query work done with a timestamp.
        4. Generate a high-temperature, surreal dream based on the query.
        5. Log the dream.
        6. End the cycle.
        """
        print("=== EonAgent Cycle Start ===")
        
        # Step 1: Report the last dream
        last_dream = self.get_last_dream()
        print("[Eon] Last Dream:", last_dream)
        
        # Step 2: Process the query and execute necessary functionality
        result = self.execute_query(query)
        print("[Eon] Output:", result)
        
        # Step 3: Log the query work done with a timestamp
        self.log_query(query, result)
        
        # Step 4: Generate a high-temperature, surreal dream based on the query
        dream = self.generate_dream(query)
        print("[Eon] Dreams:", dream)
        
        # Step 5: Log the dream separately
        self.log_dream(dream)
        
        print("=== EonAgent Cycle Complete ===\n")


# =======================
# Example Usage: Run One Cycle with a Given Query
# =======================
if __name__ == "__main__":
    agent = EonAgent()
    sample_query = "Generate a report on gene expression levels"
    agent.run_cycle(sample_query)
