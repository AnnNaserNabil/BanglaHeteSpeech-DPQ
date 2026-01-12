import torch
import torch.nn as nn
import copy

def test_deepcopy_saving():
    # Create a simple model
    model = nn.Linear(10, 1)
    
    # Save initial state with deepcopy
    saved_state = copy.deepcopy(model.state_dict())
    
    # Update model weights
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)
    
    # Check if saved state is different from current state
    current_state = model.state_dict()
    
    # Verify weights are different
    weights_match = torch.equal(saved_state['weight'], current_state['weight'])
    bias_match = torch.equal(saved_state['bias'], current_state['bias'])
    
    print(f"Weights match: {weights_match}")
    print(f"Bias match: {bias_match}")
    
    if not weights_match and not bias_match:
        print("✅ SUCCESS: Saved state is independent of model updates.")
    else:
        print("❌ FAILURE: Saved state was updated along with the model.")

def test_reference_saving_failure():
    # This demonstrates the bug we fixed
    model = nn.Linear(10, 1)
    
    # Save initial state WITHOUT deepcopy (just reference)
    saved_state = model.state_dict()
    
    # Update model weights
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)
    
    # Check if saved state is same as current state
    current_state = model.state_dict()
    
    # Verify weights are SAME (this is the bug)
    weights_match = torch.equal(saved_state['weight'], current_state['weight'])
    bias_match = torch.equal(saved_state['bias'], current_state['bias'])
    
    print(f"\n[Bug Demo] Weights match: {weights_match}")
    print(f"[Bug Demo] Bias match: {bias_match}")
    
    if weights_match and bias_match:
        print("✅ Bug confirmed: Reference saving causes saved state to change.")

if __name__ == "__main__":
    test_deepcopy_saving()
    test_reference_saving_failure()
