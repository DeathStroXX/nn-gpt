import torch
import torch.nn as nn
import random
import textwrap

# --- 1. SEARCH SPACE (Massive Diversity Expansion) ---
SEARCH_SPACE = {
    'n_columns': [2, 3],               
    'base_channels': [16, 32, 64], 
    'dropout_prob': [0.0, 0.1, 0.2, 0.3],
    'lr':[0.01, 0.005, 0.003, 0.002, 0.001],
    'momentum':[0.75, 0.8, 0.85, 0.9, 0.92, 0.95],
    'n_blocks': [2, 3],                
    
    # Structural Diversity
    'activation': ['ReLU', 'GELU', 'LeakyReLU', 'SiLU'],
    'kernel_size':[3, 5],
    'pooling_type': ['Max', 'Avg'],
    
    # Advanced Diversity Genes
    'conv_type': ['Standard', 'Depthwise'],     
    'norm_type': ['BatchNorm', 'InstanceNorm'], 
    'optimizer_type':['SGD', 'AdamW', 'RMSprop'], 
    'fc_dropout':[0.0, 0.2, 0.5]               
}

def create_random_chromosome():
    return {k: random.choice(v) for k, v in SEARCH_SPACE.items()}

# --- 2. CODE GENERATOR ---
def generate_model_code_string(chromosome: dict) -> str:
    # 1. Initial Python-side safety casting
    cols = int(float(chromosome.get('n_columns', 2)))
    chan = int(float(chromosome.get('base_channels', 32)))
    n_blocks = int(float(chromosome.get('n_blocks', 2)))
    drop = float(chromosome.get('dropout_prob', 0.1))
    lr = float(chromosome.get('lr', 0.01))
    mom = float(chromosome.get('momentum', 0.9))
    
    # Standard Structural Genes
    act_func = str(chromosome.get('activation', 'ReLU'))
    k_size = int(float(chromosome.get('kernel_size', 3)))
    pool_type = str(chromosome.get('pooling_type', 'Max'))
    
    # Advanced Genes
    conv_type = str(chromosome.get('conv_type', 'Standard'))
    norm_type = str(chromosome.get('norm_type', 'BatchNorm'))
    opt_type = str(chromosome.get('optimizer_type', 'SGD'))
    fc_drop = float(chromosome.get('fc_dropout', 0.0))
    
    # Calculate padding to keep spatial dimensions identical
    pad_size = k_size // 2 

    # 2. PyTorch String Generation
    code = textwrap.dedent(f"""
        import torch
        import torch.nn as nn
        from typing import List

        # --- HASH IDENTIFIERS (Ensures unique UUIDs for caching) ---
        # LR: {lr}
        # Momentum: {mom}
        # Activation: {act_func}
        # Kernel: {k_size}
        # Pooling: {pool_type}
        # Conv Type: {conv_type}
        # Norm Type: {norm_type}
        # Optimizer: {opt_type}
        # FC Dropout: {fc_drop}

        # --- MANDATORY FOR EVAL ENGINE ---
        def supported_hyperparameters():
            return {{'lr', 'momentum'}}

        # --- Helper Classes ---
        class FractalDropPath(nn.Module):
            def __init__(self, drop_prob: float = {drop}):
                super().__init__()
                self.drop_prob = drop_prob
            
            def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
                if not self.training: 
                    return torch.stack(inputs).mean(dim=0)
                n = len(inputs)
                mask = torch.bernoulli(torch.full((n,), 1 - self.drop_prob, device=inputs[0].device))
                if mask.sum() == 0: 
                    mask[torch.randint(0, n, (1,)).item()] = 1.0
                active =[inp for inp, m in zip(inputs, mask) if m > 0]
                return torch.stack(active).mean(dim=0)

        class FractalBlock(nn.Module):
            def __init__(self, n_columns: int, channels: int, dropout_prob: float):
                super().__init__()
                self.n_columns = int(n_columns)
                channels = int(channels)  
                
                # 1. Dynamic Activations
                if "{act_func}" == "GELU":
                    activation_layer = nn.GELU()
                elif "{act_func}" == "LeakyReLU":
                    activation_layer = nn.LeakyReLU(inplace=True)
                elif "{act_func}" == "SiLU":
                    activation_layer = nn.SiLU(inplace=True)
                else:
                    activation_layer = nn.ReLU(inplace=True)

                # 2. Dynamic Convolution Strategy
                if "{conv_type}" == "Depthwise":
                    conv_layer = nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size={k_size}, padding={pad_size}, groups=channels, bias=False),
                        nn.Conv2d(channels, channels, kernel_size=1, bias=False)
                    )
                else:
                    conv_layer = nn.Conv2d(channels, channels, kernel_size={k_size}, padding={pad_size}, bias=False)

                # 3. Dynamic Normalization Strategy
                if "{norm_type}" == "InstanceNorm":
                    norm_layer = nn.InstanceNorm2d(channels, affine=True)
                else:
                    norm_layer = nn.BatchNorm2d(channels)

                # Assemble Convolutional Sequence
                self.conv = nn.Sequential(
                    conv_layer,
                    norm_layer,
                    activation_layer
                )

                if self.n_columns > 1:
                    self.left = FractalBlock(self.n_columns - 1, channels, dropout_prob)
                    self.right_1 = FractalBlock(self.n_columns - 1, channels, dropout_prob)
                    self.right_2 = FractalBlock(self.n_columns - 1, channels, dropout_prob)
                    self.join = FractalDropPath(drop_prob=dropout_prob)

            def forward(self, x):
                if self.n_columns == 1: return self.conv(x)
                out_left = self.left(x)
                out_right = self.right_2(self.right_1(x))
                return self.join([out_left, out_right])

        # --- Main Network ---
        class Net(nn.Module):
            def __init__(self, in_shape, out_shape, prm, device):
                super(Net, self).__init__()
                self.device = device

                c_in = 3 
                n_classes = out_shape[0] if out_shape else 10
                start_chan = int({chan})  

                self.entry = nn.Sequential(
                    nn.Conv2d(c_in, start_chan, kernel_size=3, padding=1),
                    nn.BatchNorm2d(start_chan),
                    nn.ReLU(inplace=True)
                )

                blocks =[]
                pools =[]
                trans_layers =[]
                cur_chan = start_chan
                total_blocks = int({n_blocks})
                
                for i in range(total_blocks):
                    blocks.append(FractalBlock(int({cols}), cur_chan, {drop}))
                    
                    if "{pool_type}" == "Avg":
                        pools.append(nn.AvgPool2d(2))
                    else:
                        pools.append(nn.MaxPool2d(2))
                    
                    if i < total_blocks - 1:
                        next_chan = int(cur_chan * 2) 
                        trans_layers.append(nn.Sequential(
                            nn.Conv2d(cur_chan, next_chan, kernel_size=1),
                            nn.BatchNorm2d(next_chan),
                            nn.ReLU(inplace=True)
                        ))
                        cur_chan = next_chan
                    else:
                        trans_layers.append(None)

                self.blocks = nn.ModuleList(blocks)
                self.pools = nn.ModuleList(pools)
                self.trans_layers = nn.ModuleList([t for t in trans_layers if t is not None])
                self.final_channels = cur_chan

                self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
                
                # Regularization before classifier head
                self.fc_dropout = nn.Dropout(p={fc_drop})
                self.fc = nn.Linear(self.final_channels, n_classes)
                self.to(device)

            def forward(self, x):
                x = self.entry(x)
                t_idx = 0
                for i, (block, pool) in enumerate(zip(self.blocks, self.pools)):
                    x = block(x)
                    x = pool(x)
                    if i < len(self.trans_layers):
                        x = self.trans_layers[t_idx](x)
                        t_idx += 1
                x = self.global_pool(x)
                x = x.flatten(1)
                
                x = self.fc_dropout(x)
                x = self.fc(x)
                return x

            def train_setup(self, prm):
                self.criterion = nn.CrossEntropyLoss()
                
                # Dynamic Optimizer Selection
                if "{opt_type}" == "AdamW":
                    self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm['lr'], weight_decay=1e-4)
                elif "{opt_type}" == "RMSprop":
                    self.optimizer = torch.optim.RMSprop(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
                else:
                    self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

                self.max_batches = prm.get('max_batches', None)
                return self.optimizer

            def learn(self, train_data):
                self.train()
                for i, (inputs, labels) in enumerate(train_data):
                    if self.max_batches is not None and i >= self.max_batches: break
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
                    self.optimizer.step()
    """)
    return code.strip()