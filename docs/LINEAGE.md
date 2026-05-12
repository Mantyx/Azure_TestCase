# Data Lineage

Full traceability chain from Bronze to model.

## Pipeline Flow

```
NAS  →  Bronze  →  Silver  →  Gold  →  Azure ML  →  Model Registry
         (raw)   (queryable)  (training-ready)
```

## Traceability Questions

- Which annotator labeled a specific frame?
- What raw Bronze file produced a Silver row?
- What query selected the training frames?
- What code, data, and hardware produced a given model version?
