import streamlit as st
import requests
import json

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"


SYSTEM_PROMPT_DEFAULT = ""  # Empty system prompt


def generate_llama_response_default(prompt, conversation_history):
    system_prompt = SYSTEM_PROMPT_DEFAULT
    return generate_llama_response_stream(prompt, conversation_history, system_prompt)


SYSTEM_PROMPT_STRUCTURE_GUIDANCE = """
You are a highly skilled assistant specialized in writing SystemVerilog and Verilog code. 
Please structure your response in the following way:
1. Write the pseudocode first.
2. Include comments explaining each major step.
3. End with a basic testbench if applicable.
Please write clean, modular, and reusable code.
"""

def generate_llama_response_structure(prompt, conversation_history):
    system_prompt = SYSTEM_PROMPT_STRUCTURE_GUIDANCE
    return generate_llama_response_stream(prompt, conversation_history, system_prompt)


SYSTEM_PROMPT_BEST_PRACTICES = """
You are an expert in SystemVerilog and Verilog code with a deep understanding of best practices.
Please ensure the code you generate adheres to industry standards and best practices. 
This includes proper naming conventions, modular design, and thorough documentation with comments.
Start with pseudocode, then write the actual code.
"""

def generate_llama_response_best_practices(prompt, conversation_history):
    system_prompt = SYSTEM_PROMPT_BEST_PRACTICES
    return generate_llama_response_stream(prompt, conversation_history, system_prompt)


SYSTEM_PROMPT_CHAIN_OF_THOUGHT = """
You are a highly skilled assistant specialized in writing SystemVerilog and Verilog code.
Before writing the code, think through the problem step by step.
Start by identifying the key requirements and constraints, then outline a plan, and finally write the pseudocode followed by the actual code.
Your process should look like this:
1. Identify the key requirements.
2. Outline the plan for implementation.
3. Write the pseudocode.
4. Provide the final SystemVerilog code.
"""


def generate_llama_response_chain_of_thought(prompt, conversation_history):
    system_prompt = SYSTEM_PROMPT_CHAIN_OF_THOUGHT
    return generate_llama_response_stream(prompt, conversation_history, system_prompt)



SYSTEM_PROMPT_OPTIMIZATION = """
You are a seasoned SystemVerilog and Verilog developer specializing in writing highly optimized and efficient code.
Your code should prioritize performance, minimize resource usage, and consider timing constraints.
Please begin with pseudocode and follow with the optimized implementation.
"""

def generate_llama_response_optimization(prompt, conversation_history):
    system_prompt = SYSTEM_PROMPT_OPTIMIZATION
    return generate_llama_response_stream(prompt, conversation_history, system_prompt)


SYSTEM_PROMPT_USE_CASE = """
You are an expert in SystemVerilog and Verilog, well-versed in hardware design for specific applications.
Please write code tailored for the following use case: {minimal code generation}.
Begin with pseudocode and then provide the detailed implementation, including any necessary modules and testbenches.
"""

def generate_llama_response_use_case(prompt, conversation_history, use_case):
    system_prompt = SYSTEM_PROMPT_USE_CASE.format(use_case=use_case)
    return generate_llama_response_stream(prompt, conversation_history, system_prompt)


def generate_llama_response_stream(prompt, conversation_history, system_prompt):
    # Prepare the full conversation context with the provided system prompt
    full_prompt = f"System: {system_prompt}\n" + "\n".join(conversation_history + [f"Human: {prompt}", "Assistant:"])
    
    # Prepare the request payload
    payload = {
        "model": "llama3",
        "prompt": full_prompt,
        "stream": True
    }

    # Send request to Ollama API
    with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get('response', '')
        else:
            yield f"Error: Unable to generate response. Status code: {response.status_code}"


def main():
    st.title("Chat with Llama 3")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Prompting strategy selection
    strategy = st.selectbox("Choose a prompting strategy:", [
        "Default (No Guidance)", 
        "Structure Guidance", 
        "Best Practices", 
        "Optimization", 
        "Specific Use Case", 
        "Chain of Thought"
    ])
    use_case = st.text_input("Specify the use case (if applicable):") if strategy == "Specific Use Case" else ""

    # Reset button
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        conversation_history = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages[:-1]]

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Call the appropriate response generation function
            if strategy == "Default (No Guidance)":
                response_generator = generate_llama_response_default(prompt, conversation_history)
            elif strategy == "Structure Guidance":
                response_generator = generate_llama_response_structure(prompt, conversation_history)
            elif strategy == "Best Practices":
                response_generator = generate_llama_response_best_practices(prompt, conversation_history)
            elif strategy == "Optimization":
                response_generator = generate_llama_response_optimization(prompt, conversation_history)
            elif strategy == "Specific Use Case":
                response_generator = generate_llama_response_use_case(prompt, conversation_history, use_case)
            elif strategy == "Chain of Thought":
                response_generator = generate_llama_response_chain_of_thought(prompt, conversation_history)

            for response_chunk in response_generator:
                full_response += response_chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
