from transformers import AutoModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Carregue o modelo e o tokenizer
model_name = "microsoft/table-transformer"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#optimzer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs = batch['image']
        labels = batch['labels']
        outputs = model(inputs, labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
