# EXP-0018 listening sheet (v1)

Status: **not yet accepted by a human listener**. Every row below is unscored until the owner listens; no claim about naturalness may be made before that.

Previews: /Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-pairs/v1 (outside Git; the real half is licensed audio).
Each folder holds real.wav, mechanical.wav, diagnostics.png (real waveform, F0: green=real / cyan=mechanical / red=rejected frames, real spectrogram, mechanical spectrogram) and notes.json.

Rate each row 1-5 (1 unusable, 5 indistinguishable) or write a comment; empty means untested.

| id | utt | speaker | gender | pinyin | template | ready | timbre | naturalness | content | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_01 | SSB03160415 | SSB0316 | male | wo3 zai4 na4 de5 shi2 hou4 | yin | no | | | | |
| sample_02 | SSB10080285 | SSB1008 | female | you3 qian2 zhen1 xing4 fu2 | yang | no | | | | |
| sample_03 | SSB00180288 | SSB0018 | female | wo3 jiu4 bang1 ni3 tao2 ming4 | yang | no | | | | |
| sample_04 | SSB04350385 | SSB0435 | female | su1 san1 de5 lian4 wu3 gong1 | yang | no | | | | |
| sample_05 | SSB11380340 | SSB1138 | female | bei1 ai1 | yang | no | | | | |
| sample_06 | SSB09130026 | SSB0913 | female | ku4 pai4 | yang | no | | | | |
| sample_07 | SSB14520481 | SSB1452 | female | gong1 gong4 ce4 suo3 | yang | no | | | | |
| sample_08 | SSB05440115 | SSB0544 | male | jing4 hai3 | yin | no | | | | |
| sample_09 | SSB02880339 | SSB0288 | female | du4 qing2 | yin | no | | | | |
| sample_10 | SSB18630221 | SSB1863 | male | tao2 pao3 | yin | no | | | | |
| sample_11 | SSB18910490 | SSB1891 | female | du2 ni3 | yin | no | | | | |
| sample_12 | SSB06010331 | SSB0601 | female | bai2 bing1 | yang | no | | | | |
| sample_13 | SSB03950362 | SSB0395 | female | ti2 shi4 | yin | no | | | | |
| sample_14 | SSB13770005 | SSB1377 | female | bu4 xing2 ma5 | yang | no | | | | |
| sample_15 | SSB18370358 | SSB1837 | female | le4 gou4 | yin | no | | | | |
| sample_16 | SSB15930434 | SSB1593 | female | li2 ming2 | yang | no | | | | |
| sample_17 | SSB03390129 | SSB0339 | female | dian3 ge1 | yang | no | | | | |
| sample_18 | SSB07580481 | SSB0758 | female | dan4 bao3 | yang | no | | | | |
| sample_19 | SSB01450094 | SSB0145 | female | qing2 ge1 | yang | no | | | | |
| sample_20 | SSB12740255 | SSB1274 | female | zhong1 guo2 mei3 | yang | no | | | | |

## Failure reasons recorded for each sample

- sample_01 SSB03160415: status_blocked, error_issues, low_boundary_confidence, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_02 SSB10080285: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_03 SSB00180288: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_04 SSB04350385: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_05 SSB11380340: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_06 SSB09130026: status_rejected, error_issues, no_alignment, no_pitch_frames; issues: mostly_unvoiced, resampled
- sample_07 SSB14520481: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_08 SSB05440115: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_09 SSB02880339: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_10 SSB18630221: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_11 SSB18910490: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_12 SSB06010331: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_13 SSB03950362: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_14 SSB13770005: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_15 SSB18370358: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_16 SSB15930434: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_17 SSB03390129: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_18 SSB07580481: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_19 SSB01450094: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
- sample_20 SSB12740255: status_blocked, error_issues, segmentation_count_disagrees, no_pitch_frames; issues: backend_inventory_insufficient, resampled, syllable_count_mismatch
