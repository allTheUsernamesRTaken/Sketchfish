# `data/` — de-risk dataset (gitignored)

This directory holds the hand-collected images for the **M2 detection de-risk** experiment
(see `detection_report.md`). Everything here except the `.md` reports, `sources.csv`, and
`results.csv` is gitignored per `AGENTS.md` ("never commit datasets").

## Contents
```
line_art/      15 curated real portrait line drawings  (la*.jpg/png)
photos/        real face photographs                    (ph*.jpg)
xdog/          XDoG sketch-style conversions of 5 photos (ph*_xdog.png)
overlays/      detection overlays (red = landmarks, green = landmark bbox)
_scripts/      collection + experiment code (collect*.py, run_facemesh.py, …)
sources.csv    the 20 images actually used, with source URL + license
results.csv    raw per-image detection results
```

## Provenance
All images were downloaded from **Wikimedia Commons** (public-domain or CC-licensed) via its
search API; see `sources.csv` for the exact source page URL and license of every image used,
and `_scripts/manifest.json` for the full download log (including discarded candidates).

> **Licensing / curation note.** Commons skews toward *skilled, published* art (master
> drawings, engravings, ukiyo-e). True novice phone-photographed sketches — the real product
> input — are under-represented, so the measured hit rate is an **optimistic ceiling**.
> 20 candidates that turned out to be full-figure scenes, paintings, text pages, photos, or
> duplicates were downloaded then discarded by eye; only the 20 in `sources.csv` were scored.

## Images used (15 line drawings + 5 photos)

| id | content | source |
|----|---------|--------|
| la05 | loose graphite sketch, man w/ hat, 3/4 | Cooper Hewitt, *Portrait of a Man*, 1869 |
| la06 | simple clean newspaper line drawing, woman, frontal | *Generic line drawing of woman*, 1910 |
| la07 | chalk/pencil drawing, man w/ wide hat | Wellcome, *H. Spiegel*, c.1793 |
| la08 | bold high-contrast ink portrait, man, frontal | *Isaac Joel Linetzky* line portrait |
| la13 | blind-contour drawing, young man (crude) | *Blind Contour Drawing…* J.D. Cabe |
| la14 | loose charcoal, boxer, head down 3/4 | H. Major, *Jack Dempsey*, c.1923 |
| la15 | graphite sketch, man, strict profile | *Gilbert Murray pencil sketch* |
| la17 | realistic shaded pencil, woman, 3/4 | *Indian woman – charcoal pencil sketch* |
| la19 | loose graphite sketch, man w/ glasses, 3/4 | *Pencil Sketch of Henry Dawson Lowry* |
| la20 | modern shaded pencil, man, frontal | *Pencil sketch of Marc Dutroux* |
| la21 | faint light pencil sketch, bearded man | Tee-Van, *Rabindranath Tagore*, 1916 |
| la25 | stipple engraving, old woman, 3/4 | Wellcome, *Head of an old man* ["Age"] |
| la28 | ukiyo-e woodblock, woman, stylized | Utamaro, *Ase o fuku onna* |
| la30 | colored-chalk drawing, woman, profile | Rossetti, *Alexa Wilding study*, 1870 |
| la32 | soft graphite drawing, young woman | Sandys, *Head of a young woman* |
| ph08 | modern color studio headshot, man, frontal | *Aronberg headshot* |
| ph10 | official portrait, older man w/ glasses | *Pranab Mukherjee Portrait* |
| ph11 | elderly woman close-up, frontal | *Elderly Gambian woman face portrait* |
| ph15 | woman w/ facial tattoos, frontal | *Kutia Kondh woman* |
| ph18 | woman w/ cigarette, frontal | *Woman with hand-rolled cigarette* |

Full URLs + licenses: `sources.csv`.
