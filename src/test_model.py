from model import get_model
import torch

model = get_model()

print(model)

# Teste com um batch falso
dummy_input = torch.randn(16, 3, 224, 224)
output = model(dummy_input)

print("Formato da saída:", output.shape)