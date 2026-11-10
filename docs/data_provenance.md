# Data Provenance

Last updated: 2026-11-01. No dataset file is committed to git. All downloads live in `data/raw/`, which `.gitignore` excludes.

## Breakfast Actions

| Field | Value |
|---|---|
| License | Creative Commons Attribution 4.0 (CC BY 4.0), as stated on the Hugging Face dataset card. The lab page itself shows only "© 2026 Serre Lab. All rights reserved." and no license text. |
| Lab page | https://serre.lab.brown.edu/breakfast-actions-dataset.html (accessed 2026-11-01) |
| Annotation source | https://huggingface.co/datasets/Serrelab/breakfast-actions (official lab mirror, linked from the lab page) |
| Video source | Google Drive file `1jgSoof1AatiDRpGY091qd4TEKF-BUt6I` (`BreakfastII_15fps_qvga_sync.tar.gz`, 3,930,212,562 bytes), linked from the mirror's card. The card reported a quota block on 2026-10-11; the download worked on 2026-11-01. |
| Video format | 320×240, 15 fps; 3–5 uncalibrated cameras per kitchen; 52 persons; 18 kitchens; ~77 h |
| Splits | s1 = P03–P15, s2 = P16–P28, s3 = P29–P41, s4 = P42–P54 (by person) |

Files downloaded and verified:

| File | Bytes | SHA-256 | Check |
|---|---:|---|---|
| `segmentation_coarse.tar.gz` | 291,172 | `b9929b596a84f92622c2bf690b24b68ccf399cc7edd7254ac3e3fb0848217945` | matches `SHA256SUMS` |
| `segmentation_fine.tar.gz` | 931,312 | `1ab3d69ce2888944addd37f4cf5178068f791986248c45ff3aca0fad740e4b6b` | matches `SHA256SUMS` |
| `BreakfastII_15fps_qvga_sync.tar.gz` | 3,930,212,562 | `6d1cf94c225c3edeacf10116a34669f7e688dd89e1525422950b478383c0a420` (computed locally; no published checksum) | size matches the source; gzip integrity test passed (2026-11-01) |

Citation:

```bibtex
@inproceedings{kuehne2014language,
  title={The language of actions: Recovering the syntax and semantics of goal-directed human activities},
  author={Kuehne, Hilde and Arslan, Ali and Serre, Thomas},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2014}
}
@inproceedings{kuehne2016end,
  title={An end-to-end generative framework for video segmentation and recognition},
  author={Kuehne, Hilde and Gall, Juergen and Serre, Thomas},
  booktitle={IEEE Winter Conference on Applications of Computer Vision},
  year={2016}
}
```

## 50 Salads (blocked)

| Field | Value |
|---|---|
| License | CC BY-NC-SA 4.0 (University of Dundee research portal, DOI 10.15132/10000120). Acceptable for a noncommercial test set; never redistribute. |
| Status on 2026-11-01 | The data server `cvip.computing.dundee.ac.uk` does not resolve. The portal's ReadMe PDF returns a bot-check page. |
| Third-party copy | `dinggd/50salads` on Hugging Face (4.17 GB zip, no license on its card). It holds precomputed features for action segmentation, not raw video, so it cannot test our encoder. Not used. |
| Next check | Try the official server again before Phase 2a evaluation. |

## Rules

- Never commit, mirror, or redistribute video.
- Exception, approved by the user on 2026-11-09: `docs/how_it_works.html` (the visual explainer) shows 26 still frames from Breakfast Actions at 320×240 or smaller, with the citation, under the CC BY 4.0 license on the dataset card. No video is committed.
- Keep the citation in the README and in any published result.
- Record every new download here with its source, size, and checksum.
