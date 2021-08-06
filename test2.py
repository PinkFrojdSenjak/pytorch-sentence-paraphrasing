import torch
mask = torch.nn.Transformer.generate_square_subsequent_mask(4,4)

print(mask)
