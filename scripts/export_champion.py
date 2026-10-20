"""Export the registry's champion model to models/champion/ for container serving."""

from retailpulse.config import PROJECT_ROOT
from retailpulse.forecasting.serve import export_champion

if __name__ == "__main__":
    out = PROJECT_ROOT / "models" / "champion"
    version = export_champion(out)
    print(f"Exported champion v{version} to {out}")
