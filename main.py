import ollama

stream = ollama.chat(
    model='llama3',
    messages=[{'role': 'user', 'content': 'Write SystemVerilog code for 8-bit full adder.'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)