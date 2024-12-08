import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List

class AspectExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(
            config.HIDDEN_SIZE,
            config.HIDDEN_SIZE//2,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(config.HIDDEN_SIZE, 3)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        lstm_out, _ = self.lstm(sequence_output)
        logits = self.classifier(lstm_out)
        return logits
    
    def extract_aspects(self, texts: List[str], tokenizer) -> List[str]:
        self.eval()
        aspects = []
        
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.MAX_SEQUENCE_LENGTH
            )
            
            with torch.no_grad():
                logits = self(inputs['input_ids'], inputs['attention_mask'])
                
            predictions = torch.argmax(logits, dim=-1)
            aspect_tokens = inputs['input_ids'][predictions == 0]
            aspects.extend(tokenizer.decode(token) for token in aspect_tokens)
            
        return list(set(aspects))
    
    def train(self, train_loader, val_loader, num_epochs=10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            self.train()
            total_train_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = self(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, 3), labels.view(-1))
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = self(input_ids, attention_mask)
                    loss = criterion(outputs.view(-1, 3), labels.view(-1))
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Average validation loss: {avg_val_loss:.4f}')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.state_dict(), 'best_aspect_extractor.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping triggered')
                    break
    
    