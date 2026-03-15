from pathlib import Path

from Jabberjay import Dataset, Jabberjay, Model, Visualisation

RES = Path(__file__).parent.parent / "res"

jj = Jabberjay()

# Discover available options
jj.list_models()
jj.list_datasets()
jj.list_visualisations()


def run_all_models(audio_path: Path) -> None:
    # Classical and RawNet2 — no dataset or visualisation required
    for model in ("Classical", "RawNet2"):
        result = jj.detect(audio_path, model=model)
        print(f"[{model}] {result}")

    # VIT — requires dataset and visualisation
    for vis in Visualisation:
        for ds in Dataset:
            result = jj.detect(
                audio_path,
                model=Model.VIT,
                visualisation=vis,
                dataset=ds,
            )
            print(f"[VIT/{vis.value}/{ds.value}] {result}")

    # AST — requires dataset
    for ds in Dataset:
        result = jj.detect(audio_path, model=Model.AST, dataset=ds)
        print(f"[AST/{ds.value}] {result}")


print("\n--- Bonafide ---")
run_all_models(RES / "bonafide" / "bonafide.flac")

print("\n--- Spoof ---")
run_all_models(RES / "spoof" / "spoof.flac")
