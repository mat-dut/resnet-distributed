# ResNet-50: sekwencyjnie, równolegle, rozproszenie

Rozpoznawanie obrazów modelem ResNet-50 w trzech wersjach:

- **sekwencyjna** (`sequential.py`): 1 proces, 1 wątek, 1 obraz naraz,
- **równoległa** (`parallel.py`): model podzielony na 4 etapy, każdy w osobnym procesie, połączone kolejkami,
- **rozproszona** (`agent.py`): laptopy (agenci) pobierają paczki plików do przetworzenia i odsyłają wyniki.

## Przygotowanie

Wymagany jest [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run prepare_data.py
```

`prepare_data.py` pobiera 1000 przykładowych obrazów ImageNet i tworzy `data/bench10k` z 10 000 plików.

## Wersja sekwencyjna i równoległa

```bash
uv run sequential.py data/bench10k --limit 1000 -o results_seq.csv
uv run parallel.py   data/bench10k --limit 1000 -o results_par.csv
```

Na końcu każdy program wypisuje czas i liczbę obrazów na sekundę. Opcja `--limit` ogranicza liczbę obrazów.

## Wersja rozproszona

Na każdym laptopie, po wykonaniu kroków z „Przygotowania”:

```bash
uv run agent.py http://<IP_SERWERA>:8000 data/bench10k --name <nazwa>
```
