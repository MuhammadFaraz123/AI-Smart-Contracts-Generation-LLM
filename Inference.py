import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.optim as optim

device = ("cuda" if torch.cuda.is_available() else "cpu" )

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("")
model = AutoModelForCausalLM.from_pretrained("").to(device)

def generate_response(prompt):
  
    input_ids = tokenizer.encode(to_qry(prompt),return_tensors='pt').to(device)
    input_ids = input_ids
    
    sample_outputs = model.generate(
        input_ids,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True, 
        max_length=1400, 
        top_k=3,    
        temperature = 0.02,
        num_return_sequences=1
            
    )

    res = ""
    for i, sample_output in enumerate(sample_outputs):
        res = tokenizer2.decode(sample_output, skip_special_tokens=True).split('Response:')[-1]
        

    return res
    
def to_qry(prompt):
    list_data = f'Below is an instruaction that describes a task. Write a response that appropriately completes the request. \nInstruction:{prompt} \nResponse:'
    return list_data
    
qry = "Develop a governance token in Solidity that enables holders to participate in on-chain governance. Implement a voting mechanism where token holders can propose and vote on governance proposals."

print(generate_response(to_qry(qry)))
