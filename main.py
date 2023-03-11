import torch
from speech_transformer import SpeechTransformer

BATCH_SIZE, SEQ_LENGTH, DIM, NUM_CLASSES = 3, 12345, 80, 4

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print(device)

inputs = torch.rand(BATCH_SIZE, SEQ_LENGTH, DIM).to(device)
input_lengths = torch.IntTensor([100, 50, 8])
targets = torch.LongTensor([[2, 3, 3, 3, 3, 3, 2, 2, 1, 0],
                            [2, 3, 3, 3, 3, 3, 2, 1, 2, 0],
                            [2, 3, 3, 3, 3, 3, 2, 2, 0, 1]]).to(device)  # 1 means <eos_token>
target_lengths = torch.IntTensor([10, 9, 8])

model = SpeechTransformer(num_classes=NUM_CLASSES, d_model=512, num_heads=8, input_dim=DIM)
predictions, logits = model(inputs, input_lengths, targets, target_lengths)
