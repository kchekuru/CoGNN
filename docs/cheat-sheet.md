(base) PS C:\Users\kisho\src\house\ung-dl\CoGNN> conda create -n cgnn python=3.9 -y
(base) PS C:\Users\kisho\src\house\ung-dl\CoGNN> conda activate cgnn

Open Command Palette: Ctrl+Shift+P.
Run Python: Select Interpreter.
select 'cgnn'
relaod window/workspace

or 

conda activate cgnn

run pip commands from with in the workspace [ Notice (cgnn) in the front - prefixed conda environment - not base]
(cgnn) PS C:\Users\kisho\src\house\ung-dl\CoGNN> pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
(cgnn) PS C:\Users\kisho\src\house\ung-dl\CoGNN> pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
(cgnn) PS C:\Users\kisho\src\house\ung-dl\CoGNN> pip install torch-geometric==2.3.0  
(cgnn) PS C:\Users\kisho\src\house\ung-dl\CoGNN> pip install torchmetrics ogb rdkit matplotlib tqdm numpy  

-- to verify --
python -c "import torch; print(torch.version)"
python -c "import torch_geometric; print(torch_geometric.version)"
python -c "import torch; print(torch.version, torch.version.cuda, torch.cuda.is_available())"
python -c "import torch_geometric; print(torch_geometric.version)"
python main.py --dataset pubmed --max_epochs 3 --batch_size 16