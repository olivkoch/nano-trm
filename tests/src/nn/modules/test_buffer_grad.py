# test_buffer_grad.py
import torch
import torch.nn as nn
from torch.optim import SGD

print(f"PyTorch version: {torch.__version__}")

# Test if nn.Buffer exists and how it handles requires_grad
try:
    # Test 1: Check if nn.Buffer exists
    test_tensor = torch.zeros(5, 10, requires_grad=True)
    buffer = nn.Buffer(test_tensor, persistent=False)
    print("✓ nn.Buffer exists")
    print(f"  Buffer requires_grad: {buffer.requires_grad}")
    print(f"  Buffer is_leaf: {buffer.is_leaf}")

except AttributeError as e:
    print(f"✗ nn.Buffer not found: {e}")


# Test 2: Create a minimal module with Buffer
class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Create buffer with requires_grad
        self.buffer_with_grad = nn.Buffer(torch.zeros(5, 10, requires_grad=True), persistent=False)
        self.buffer_no_grad = nn.Buffer(torch.zeros(5, 10), persistent=False)
        self.regular_param = nn.Parameter(torch.zeros(5, 10))

    def forward(self, x):
        return x + self.buffer_with_grad


# Test 3: Try to optimize buffers
module = TestModule()
print("\nModule buffers:")
for name, buf in module.named_buffers():
    print(f"  {name}: requires_grad={buf.requires_grad}, is_leaf={buf.is_leaf}")

print("\nTrying to create optimizer with buffers...")
try:
    # Try with just the buffer
    opt1 = SGD([module.buffer_with_grad], lr=0.1)
    print("✓ Can optimize buffer with requires_grad directly")
except ValueError as e:
    print(f"✗ Cannot optimize buffer directly: {e}")

try:
    # Try with buffers() method
    opt2 = SGD(module.buffers(), lr=0.1)
    print("✓ Can optimize module.buffers()")
except ValueError as e:
    print(f"✗ Cannot optimize module.buffers(): {e}")

try:
    # Try with list of buffers
    buffer_list = list(module.buffers())
    opt3 = SGD(buffer_list, lr=0.1)
    print("✓ Can optimize list(module.buffers())")
except ValueError as e:
    print(f"✗ Cannot optimize list(module.buffers()): {e}")

# Test 4: Check if gradient flows through buffer
module = TestModule()
x = torch.ones(5, 10, requires_grad=True)
out = module(x)
loss = out.sum()
loss.backward()

print("\nGradient test:")
print(f"  buffer_with_grad.grad: {module.buffer_with_grad.grad}")
print(f"  x.grad sum: {x.grad.sum().item()}")
