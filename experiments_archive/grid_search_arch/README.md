# Grid Search Arquitectura

Esta pasta serve para testar rapidamente a grelha:

- `128, 3 layers`
- `192, 3 layers`
- `128, 4 layers`
- `192, 4 layers`

Os ficheiros principais são:

- `run_grid.ps1`: corre os 4 treinos
- `summarize_grid.py`: resume os melhores `val_top1_real`, `val_top5_real` e `val_top10_real`

Todos os runs são gravados em `GNN/runs/` com nomes começados por `grid_`.
