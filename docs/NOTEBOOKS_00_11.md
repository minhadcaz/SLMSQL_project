# Kaggle: chay tung notebook 00-11, du lieu va checkpoint

Tai lieu nay mo ta **code dang co**, khong phai mot pipeline 12 buoc noi tiep. Co
hai nhanh doc lap:

- **Nghien cuu chinh: 09 -> 10 -> 11.** Generator `cycloneboy/SLM-SQL-1.5B`
  va CSC merger `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct` cua tac
  gia; LoRA chi hoc tren merger. Khong can chay 00-08, khong can SynSQL.
- **Nhanh cu Qwen/adaptive: 00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08**
  (nhieu buoc tuy chon). 02 do Qwen base, 03 huan luyen LoRA Qwen; 04-08
  khong do CSC merger cua tac gia. Bao cao 08 khong doc report cua 09-11.

## Truoc khi mo notebook

1. Kaggle Settings: bat GPU va Internet cho nhung cell can tai weights/cai goi.
   Python khuyen nghi 3.11/3.12; 09 va 11 dung
   `requirements-csc-inference.txt`, 10 dung `requirements-csc-training.txt`
   trong **session rieng**. Khong cai hai profile vao cung mot session.
2. Clone source moi tu GitHub sau khi push; hoac attach source da duoc cap nhat.
   Dat `SLM_PRJ_PATH=/kaggle/working/SLM-PRJ` neu repo o vi tri khac. Kiem tra:

   ```python
   from pathlib import Path
   repo = Path('/kaggle/working/SLM-PRJ')
   assert (repo / 'research/slm_sql/__main__.py').is_file()
   ```

3. Kaggle Input la **read-only**. Xem dung ten thu muc dataset da attach,
   khong doan slug hoac cap thu muc:

   ```python
   from pathlib import Path
   for root in sorted(Path('/kaggle/input').iterdir()):
       print(root, [p.name for p in root.iterdir()][:8])
   ```

4. BIRD **train** can mot thu muc co `train.json` va
   `train_databases/<db_id>/<db_id>.sqlite`; **dev** can `dev.json` va
   `dev_databases/<db_id>/<db_id>.sqlite`. Path env tro toi **thu muc chua
   JSON**, khong tro toi file JSON va khong tro toi cha cua thu muc do.
   Dung `Path(..., 'train.json').is_file()` / `Path(..., 'dev.json').is_file()`
   truoc khi chay prepare. Dataset BIRD mini-dev khong thay the full dev.
5. Chi `09` tao `results/csc_run_config.json` (neu chua co). Sao luu ca JSON
   config va **nguyen thu muc** `CSC_EXPERIMENT_DIR` giua cac session; folder
   `results/` bi `.gitignore` loai khoi GitHub. Nen dat experiment trong
   `/kaggle/working`, sau do Save Version / dua output thanh Kaggle Dataset
   private. Session sau mount output read-only tai `/kaggle/input/...`, copy
   **ca folder va config** ve **dung duong dan** `/kaggle/working/...` truoc
   khi bat dau 10/11. Vi du (thay slug do Kaggle cung cap):

   ```python
   from pathlib import Path
   import shutil
   mount = Path('/kaggle/input/<saved-run>')
   configs = list(mount.rglob('csc_run_config.json'))
   assert len(configs) == 1, configs  # choose the right run when >1
   source = configs[0].parent
   target = Path('/kaggle/working/SLM-PRJ/results')
   # Chi chay mot lan tren session moi; khong de len output cua run khac.
   shutil.copytree(source / 'csc_slm_sql_pilot', target / 'csc_slm_sql_pilot')
   shutil.copy2(source / 'csc_run_config.json', target / 'csc_run_config.json')
   ```

   Vi tri that tren mounted dataset co the them mot thu muc; xem bang lenh
   liet ke truoc khi copy. `csc_run_config.json` chua path tuyet doi cua train,
   dev, experiment; mount lai cac DB vao cung path va kiem tra file ton tai.
   Khong sua config cu de khoi phuc, khong tron pilot voi full run.

## Nhanh cu: 00-08 (khong bat buoc cho CSC LoRA)

### 00 - `00_setup.ipynb`

- **Dau vao:** Git repo, full BIRD dev Kaggle Dataset hoac cho phep download dev;
  optional Spider va SynSQL. `BIRD_INPUT_DIR` neu duong dan dev khong duoc
  autodetect; `SYNSQL_RAW` neu co SynSQL. Cac helper o `scripts/kaggle_utils.py`.
- **Chay:** bat Internet; cell dau cai nhieu packages, bao gom Unsloth; chi bat
  `INSTALL_VLLM=1` neu notebook cu can vLLM. Cac cell sau clone repo, chuan bi
  `data/bird`, kiem tra gold va chay unit tests. **Khong chay 00 truoc 09**
  trong cung session: stack Unsloth/TRL va vLLM co the xung dot.
- **Dau ra:** `REPO_DIR`, `DATA_DIR`, `BIRD_DIR`, du lieu dev
  `data/bird/dev.json` + `data/bird/dev_databases/` trong working directory;
  session sau khong tu ton tai neu chua Save Version.

### 01 - `01_data_prep.ipynb`

- **Dau vao:** source va BIRD dev tu 00 (hoac attach dung layout), Spider dev
  tuy chon; `SYNSQL_RAW` tro toi JSONL hoac JSON-array export tu dataset
  SynSQL/SynsQL-Think neu muon huan luyen Qwen. Dung `SFT_SIZES`, `SFT_MAX_CHARS`,
  `SFT_SEED` de dieu chinh.
- **Chay:** load/kiem tra BIRD, optional Spider; thu gold-vs-gold 50 cau;
  neu khong co SynSQL thi bo qua phan tao subsets, khong co SFT dataset.
- **Dau ra:** `data/sft_subsets/sft_<N>.jsonl` (neu co SynSQL), duoc 03 dung.
  09-11 **khong dung** cac subsets nay.

### 02 - `02_baseline.ipynb`

- **Dau vao:** BIRD dev + `BASE_MODEL` (mac dinh
  `Qwen/Qwen2.5-Coder-1.5B-Instruct`); GPU va vLLM. Co the can
  `MAX_MODEL_LEN`, `MAX_OUTPUT_TOKENS`, `GEN_BATCH_SIZE` phu hop prompt dai.
- **Chay:** greedy va SC@16 tren tap dev da chon. Kiem tra context **truoc**
  generation; neu thieu VRAM giam batch/increase budget hop le.
- **Dau ra:** `results/baselines/<MODEL_KEY>/E0_greedy.json` va
  `results/baselines/<MODEL_KEY>/E1_sc16.json`. **Khong phai** author SLM-SQL + CSC;
  diem nay khong duoc so truc tiep voi EX cong bo.

### 03 - `03_lora_sft.ipynb`

- **Dau vao:** SynSQL subset cua 01 (`TRAIN_DATA`, mac dinh
  `data/sft_subsets/sft_10000.jsonl`); `configs/lora_grid.yaml`; model Qwen,
  Unsloth/TRL/GPU. `MAX_EXPERIMENTS=1` mac dinh, `RUN_FULL_GRID=1` moi mo grid.
- **Chay:** SFT LoRA Qwen, optional quick BIRD dev eval va merge checkpoint;
  `RESUME_FROM` chi khi checkpoint trainer phu hop. Cell cuoi co the
  `publish_checkpoint` neu co `CHECKPOINT_DIR`, `HF_REPO_ID`, `HF_TOKEN`.
- **Dau ra:** `results/lora/grid_results.json`, adapter theo experiment,
  `results/lora/best_checkpoint.json`, va thu muc `merged/` neu merge thanh cong.
  **Khong** dung Qwen adapter nay lam CSC adapter cua 10.

### 04 - `04_candidates.ipynb`

- **Dau vao:** `MODEL_PATH` la **thu muc merged model** cua 03 (khong phai
  LoRA adapter rieng); neu bo trong thi doc
  `results/lora/best_checkpoint.json` hoac mac dinh merged path. Can BIRD dev.
  Neu thu muc merged khong co, legacy co the tai checkpoint tu `HF_REPO_ID`.
- **Chay:** tao split DB-disjoint tu full BIRD dev truoc; `BIRD_LIMIT` chi lay pilot
  phan tang va van bao phu ca calibration lan evaluation. Sinh `N_CANDIDATES`
  (mac dinh 32), execute SQL, in oracle/SC va calibration.
- **Dau ra:** `results/candidates/candidates_dev.jsonl`, sau do
  `results/candidates/exec_dev.jsonl`; 05 doc `exec_dev.jsonl`.
  Khong dung candidate file nay trong 09-11 (prompt/model/protocol khac).

### 05 - `05_adaptive.ipynb`

- **Dau vao:** `EXEC_FILE` mac dinh `results/candidates/exec_dev.jsonl` tu 04,
  va BIRD dev SQLite. `N_PERMUTATIONS` mac dinh 100.
- **Chay:** mo phong fixed/adaptive, learned stopper, ablations va Pareto.
  Hai cell CSC huan luyen cu da bi vo hieu hoa vi chung co the dung dev gold;
  router **chi mo phong** so lan CSC, khong goi merger cua tac gia.
- **Dau ra:** `results/policies/simulation_results.json`,
  `ablation_results.json`, `results/final/pareto_report.json` va cac PNG.

### 06 - `06_business.ipynb`

- **Dau vao:** `business/business_questions.json` trong repo. Neu can cham
  model: dat `BUSINESS_BASE_MODEL` va `BUSINESS_LORA_MODEL` toi **hai thu muc
  full/merged model**, khong phai CSC adapter. GPU + dependencies cua demo.
- **Chay:** tao/kiem tra `business/business.sqlite`. `RUN_EVAL = False` la
  bien **viet cung trong cell**, phai sua thanh `True` khi da co model; cell
  Gradio la tuy chon. Khong danh gia CSC cua 09-11.
- **Dau ra:** `results/business/<variant>_seed<seed>.json` neu bat RUN_EVAL.

### 07 - `07_final_eval.ipynb`

- **Dau vao:** `FINAL_MODEL_PATH` = thu muc full/merged model; `BIRD_DATA_DIR`
  chua `dev.json`, optional `SPIDER_DATA_DIR` chua `dev.json`/`test.json` + DB,
  optional `ROBUSTNESS_DATA_DIR`; business DB duoc tao tu source. GPU.
- **Chay:** dat env `RUN_FINAL_EVAL=1` sau khi da kiem tra model/data; notebook
  dong bang protocol, dataset digest, config va seed 42/43/44 truoc khi sinh ket qua,
  sau do chay fixed/adaptive. Neu thieu gold Spider split se skip co thong bao.
- **Dau ra:** `results/final/*.json` va pairwise/robustness/error sample.
  Khong dung LoRA CSC adapter `adapter/final` lam `FINAL_MODEL_PATH`.

### 08 - `08_report.ipynb`

- **Dau vao:** `results/final/*.json` tu 05/07; khong can GPU, model hay BIRD.
- **Chay:** tong hop JSON hien co, bao cao McNemar/Pareto, kem hash provenance;
  neu khong co input thi bang rong, **khong** tu sinh diem.
- **Dau ra:** `results/final/report/summary_table.{json,csv}`,
  `mcnemar_table.json`, `pareto.json`/PNG va `provenance.json`.

## Nhanh chinh: 09-11 (SLM-SQL + CSC cua tac gia)

### Thiet lap chung (dat truoc cell bootstrap cua **moi session**)

```python
import os
os.environ['SLM_PRJ_PATH'] = '/kaggle/working/SLM-PRJ'
os.environ['CSC_CONFIG'] = '/kaggle/working/SLM-PRJ/results/csc_run_config.json'
os.environ['CSC_EXPERIMENT_DIR'] = '/kaggle/working/SLM-PRJ/results/csc_slm_sql_pilot'
os.environ['CSC_TRAIN_DIR'] = '/kaggle/input/<bird-train>/train'  # sua theo train.json that
os.environ['CSC_EVAL_DIR'] = '/kaggle/input/<bird-dev>/dev'       # sua theo dev.json that
os.environ['CSC_PILOT'] = '1'             # CHI co tac dung khi tao config lan dau
os.environ['INSTALL_CSC_DEPS'] = '1'     # cai profile phu hop neu chua co
os.environ['RUN_CSC_GPU'] = '0'          # preflight/plan truoc; '1' khi san sang
os.environ['BUILD_CSC_TRAIN_CACHE'] = '1'
os.environ['CSC_SEED'] = '42'
```

Kiem tra `CSC_TRAIN_DIR/train.json` va `CSC_EVAL_DIR/dev.json`. Kaggle Secrets
`HF_TOKEN`/`HF_REPO_ID` chi can neu upload/restore adapter. Tren session sau,
**restore config + experiment truoc**; dung lai env path chinh xac. `configure()`
bao loi neu env khac config da dong bang. `CSC_PILOT=0` cho full run phai dung
`CSC_CONFIG`/`CSC_EXPERIMENT_DIR` **moi**; mac dinh full: train 1200,
validation 200, **toan bo dev**, 16 samples train/validation, 64 samples dev,
8 merge samples. Khong danh dong diem pilot vao paper.

### 09 - `09_slm_sql_baseline.ipynb`

- **Dau vao:** full BIRD dev; train neu `BUILD_CSC_TRAIN_CACHE=1`. Hai checkpoint
  cua tac gia tren Hub se download o lan generation/merge GPU dau tien (can
  Internet); version thuc te duoc dong bang trong `models/{generator,merger}.json`.
- **Chay:** session inference. Cell install profile, `plan`, `prepare`, xem so
  cau truoc khi bat `RUN_CSC_GPU=1`. `baseline` chay generation -> execution
  -> CSC goc -> score trong process rieng. Neu chi co dev, set
  `BUILD_CSC_TRAIN_CACHE=0`: `prepare --evaluation-only`; khi co train, doi
  lai `1`, chay 09 voi cung config, khong regenerate evaluation da cache.
  Sau baseline, neu train cache bat, cell cuoi generate/execute train/validation.
- **Dau ra (tai `<experiment>`):** `data/manifest.json`, `data/*.jsonl`,
  `data/schemas/*.json`, `models/*.json`, `execution_cache.sqlite`,
  `seed42/evaluation.{generation,execution,author.csc-k4}.jsonl` va cac
  `.manifest.json`, `seed42/{train,validation}.{generation,execution}.jsonl`,
  `reports/evaluation-seed42-k4.baseline.json`. `k64` khi full config.
- **Canh bao:** 09 khong tao adapter; SC va author CSC score o day, khong co
  delta LoRA. Neu baseline CSC khong he duoc goi thi 10 se tu choi training.

### 10 - `10_csc_lora.ipynb`

- **Dau vao:** **toan bo** cache/config 09, BIRD train+dev SQLite o path goc
  (pair build kiem tra DB); author CSC evaluation baseline va train/validation
  execution cache cung seed. Session **training** rieng.
- **Chay:** restore folder, install profile training, `plan`, `pairs` (CPU),
  kiem tra `stats.train.examples`/`stats.validation.examples`; sau do moi
  `RUN_CSC_GPU=1` va cell `train`. Target la SQL draft dung tren **train**,
  LoRA tren merger cua tac gia, validation loss chon checkpoint. Neu pilot
  khong co cap JOIN/aggregate phu hop thi dung voi loi ro rang: tang gioi han
  o experiment **moi**, khong lay dev de huan luyen.
- **Dau ra:** `adaptation/{train,validation}.jsonl`,
  `adaptation/manifest.json`, `adapter/checkpoints/`, `adapter/final/` va
  `adapter/training_manifest.json` trong `<experiment>`.
- **Upload optional:** cell cuoi chi chay khi `PUBLISH_CSC_ADAPTER=1`; xem muc
  Hugging Face ben duoi. Khong can merge voi base checkpoint va khong upload
  weights cua tac gia.

### 11 - `11_csc_evaluation.ipynb`

- **Dau vao:** nguyen folder/config cua 09-10, BIRD **dev SQLite** o path cu;
  train DB raw khong can neu da co prepared train metadata/pair files.
  `adapter/final` va `adapter/training_manifest.json` co the duoc restore tu
  Hub *sau khi* copy toan bo cache khac. Session **inference** rieng.
- **Chay:** optional `RESTORE_CSC_ADAPTER=1` (doc Hub private). `CSC_K` mac
  dinh bang cached n (4 pilot / 64 full); K nho hon (vi du 16 tren cache 64)
  can chay them author merge cung K. Bat `RUN_CSC_GPU=1` de merge author/LoRA;
  `score` sau do chi can CPU. Generator cache khong duoc sinh lai.
- **Dau ra:** `seed42/evaluation.lora.csc-k<K>.jsonl` + manifest,
  `reports/evaluation-seed42-k<K>.comparison.json`. Mo JSON xem `summary`,
  `comparison.delta_ex`, CI, McNemar, so cau improved/regressed. Khong tu nhan
  EX cua paper neu prompt/schema khac.

## Day LoRA CSC len Hugging Face (chi sau notebook 10)

Hub la noi luu **model adapter**, khong thay the backup experiment. Tao tai
khoan Hugging Face va **fine-grained token co quyen write** cho model repo cua
ban. Them Kaggle Secrets `HF_TOKEN` va `HF_REPO_ID` (gia tri dang
`username/slm-sql-csc-lora-pilot`), bat Internet. **Khong** in token vao cell,
khong commit token vao Git. Ma se `create_repo(private=True)` neu chua co;
tu choi upload neu repo da ton tai nhung **public**. Kiem tra license cua
weights upstream va quyen phan phoi truoc khi can nhac public hoa.

```python
# Sau khi 10 da train xong va training_manifest.status == 'complete':
import os
os.environ['PUBLISH_CSC_ADAPTER'] = '1'
# Chay lai cell 'Optional: publish...' o cuoi notebook 10.
```

Upload gom `adapter/final/adapter_config.json`, `adapter_model.safetensors`
(va cac file trong `adapter/final`) tai **root model repo**, va
`.csc/training_manifest.json`. Khong upload gold BIRD, candidates, config hay
base model. Neu Hub bao 401/403: token thieu quyen/write, Kaggle Secret khong
duoc share cho notebook, hoac repo ID sai. `HF_REPO_ID` cua notebook 03 la repo
**khac** (merged Qwen); khong tai de dung lam CSC cua 11.

Trong session 11, truoc cell restore: mount output cua 09-10, copy config va
**toan bo** experiment ve path ban dau (gop `adaptation/` va `seed42/`); set
Secrets `HF_TOKEN`, `HF_REPO_ID`; dat `RESTORE_CSC_ADAPTER=1`; chay cell restore.
Helper tu choi adapter neu base settings/pair signature/hash khong khop hoac
file local khac nhau, khong ghi de adapter da co. Neu khong dung Hub, copy
`adapter/` kem trong backup la du. *Adapter rieng* khong the thay the cac
manifest hay evaluation candidates de so sanh paired.

## Loi hay gap va cach kiem tra

| Bieu hien | Nguyen nhan / thao tac |
|---|---|
| `ModuleNotFoundError: research.slm_sql` | Clone remote chua duoc push, hoac kernel dang dung repo cu; `git rev-parse HEAD`, kiem tra `research/slm_sql/__main__.py`, restart kernel neu vua cai deps. |
| `Missing SQLite database` | `CSC_*_DIR` tro sai muc; tim file `train.json` / `dev.json`, so khop `<db_id>.sqlite` trong `*_databases`. |
| Config/path differs, cache hash mismatch | Restore JSON config va experiment **cung duong dan tuyet doi**; DB path/size/mtime khong doi. Neu doi dataset/seed/sampling/version, tao experiment moi; khong sua manifest bang tay. |
| `No usable validation ... pairs` | Pilot qua nho hoac khong co top-2 JOIN/aggregate disagreement co dap an dung; dung full/large run rieng, khong dung dev labels. |
| `CUDA not available` / OOM | Bat GPU, check `nvidia-smi`; vLLM va PEFT can GPU, tach session inference/training; giam batch/context trong **run moi** sau khi can nhac prompt size. |
| Version mismatch (`vllm` / `torch` / `transformers`) | Resume tren environment phu hop version trong manifests; khong nang packages giua chay. Khong cai `00`/Unsloth/TRL vao session 09/11. |
| `CSC_K` lon hon so mau cache | Dung K <= `generation.evaluation_samples` trong config; K=16 chi hop le khi cache dev sinh >=16. |
| `HF_REPO_ID` / `HF_TOKEN` 401/403 | Secret sai/khong duoc share, token thieu write, repo public; khong dan token vao output. |
| `RUN_CSC_GPU=0` khong co EX | Dung: che do preflight khong sinh du doan; bat GPU sau `plan`, chi chay cac stage chua cache. |

Cuoi cung: nghien cuu chinh **chua co EX thuc nghiem tren BIRD that** neu chua
chay 09-11 tren GPU. Unit tests backend gia chi xac nhan code va giao thuc.
