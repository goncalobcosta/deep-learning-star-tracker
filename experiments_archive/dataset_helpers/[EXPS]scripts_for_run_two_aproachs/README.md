# testes_2aproachs

Scripts para correr e guardar historico das comparacoes entre:
- baseline (`guide_star + roll`)
- professor (`boresight + roll`)

## Execucao rapida

No terminal, a partir da raiz do projeto:

```bash
python tetra4/synth_dataset/testes_2aproachs/01_run_baseline.py
python tetra4/synth_dataset/testes_2aproachs/02_run_prof_exp_a.py
python tetra4/synth_dataset/testes_2aproachs/03_run_prof_exp_b.py
python tetra4/synth_dataset/testes_2aproachs/04_run_prof_exp_c.py --appear-cap-margin 500
python tetra4/synth_dataset/testes_2aproachs/06_run_prof_exp_d.py --appear-band-margin 500
```

Ou tudo de uma vez:

```bash
python tetra4/synth_dataset/testes_2aproachs/05_run_all.py --appear-cap-margin 500
python tetra4/synth_dataset/testes_2aproachs/05_run_all.py --appear-cap-margin 500 --include-exp-d --appear-band-margin 500
```

## Ficheiros de estado/historico

- `state.json`: guarda `last_baseline_run` para os scripts A/B/C usarem por defeito.
- `state.json`: guarda `last_baseline_run` para os scripts A/B/C/D usarem por defeito.
- `history.jsonl`: uma linha JSON por execucao, para rastrear o que foi corrido.

## Notas

- Todos os scripts escrevem runs em `synth_dataset/runs_tmp_validation` por defeito.
- Usa `--timelapse` se quiseres tambem gerar `sky_plots` (com parametros RA/Dec).
- Exemplo: `--timelapse --timelapse-ra-min 150 --timelapse-ra-max 180 --timelapse-dec-min 0 --timelapse-dec-max 30`.
