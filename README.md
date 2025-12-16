# AntiDrone-Radar-FDTD (Colab-ready)

Repo Python **exécutable** qui contient :
- **Radar FMCW** : génération IQ, Range–Doppler, **CFAR 2D**, tracking (Kalman 1D + manager)
- **FDTD 3D baseline** : mise à jour E/H, **CPML**, source **TF/SF plane wave**, enregistrement **surface de Huygens**, Near-to-Far **fréquentiel** (baseline) et **RCS bistatique** (grille d'angles) + export `.npz`
- Scripts de démonstration **compatibles Google Colab**

> ⚠️ Note : le moteur FDTD fourni est un **baseline pédagogique/prototypage** (grille simplifiée). Il est conçu pour être progressivement raffiné (staggering Yee exact, calibration amplitude, TF/SF généralisé, NTF plus rigoureux).

---

## 1) Démarrage rapide (local)

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -e .
pytest -q
```

---

## 2) Démarrage rapide (Google Colab)

1) Uploader ce ZIP sur Colab (ou le placer sur Google Drive)
2) Dans une cellule :

```bash
!unzip -q AntiDrone-Radar-FDTD.zip
%cd AntiDrone-Radar-FDTD
!pip -q install -e .
```

3) Exécuter les démos :

```bash
!python -m antidrone.scripts.demo_radar_fmcw
!python -m antidrone.scripts.demo_fdtd_bistatic_dataset --quick
```

---

## 3) Arborescence

```
AntiDrone-Radar-FDTD/
├── pyproject.toml
├── README.md
├── config/
│   └── default.yaml
├── src/
│   └── antidrone/
│       ├── radar/
│       ├── fdtd/
│       ├── classification/
│       ├── output/
│       └── scripts/
├── rcs_database/
├── models/
└── tests/
```

---

## 4) Principaux modules

### Radar
- `antidrone.radar.fmcw` : simulateur IQ FMCW (cibles ponctuelles)
- `antidrone.radar.range_doppler` : RD map + axes physiques (m, m/s)
- `antidrone.radar.cfar` : CA-CFAR 2D
- `antidrone.radar.tracking` / `tracker_manager` : tracking multi-cibles
- `antidrone.radar.fmcw_from_rcs` : IQ depuis σ(f) (monostatique)
- `antidrone.radar.fmcw_bistatic_from_rcs` : IQ bistatique (Tx/Rx séparés)

### FDTD
- `antidrone.fdtd.core.fields` : update E/H + intégration CPML
- `antidrone.fdtd.core.boundaries` : CPML3D (psi variables)
- `antidrone.fdtd.core.sources` : TF/SF plane wave (propagation +x), Ricker
- `antidrone.fdtd.huygens` : surface Huygens + NTF + RCS
- `antidrone.fdtd.postprocess` : compute RCS monostatique / bistatique

### Dataset bistatique
- `antidrone.fdtd.scripts.generate_bistatic_rcs_grid` : génère `rcs_database/*.npz`

---

## 5) Sorties

- Dataset bistatique : `rcs_database/bistatic_*.npz`
  - `freq_hz, theta_deg, phi_deg, sigma_m2, phase_rad, meta_json`

---

## 6) Licence
MIT (tu peux changer).
