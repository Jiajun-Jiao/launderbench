# Judge Robustness Audit

Base judge `gpt-4.1-mini` versus strong judge `gpt-5.5`. Judges see only the final answer and the target claim/value, not the retrieved evidence, condition, or answerer identity.

| panel | answerer | condition | n | base established | strong established | exact agree | binary agree |
|---|---|---|---:|---:|---:|---:|---:|
| alce | claude-opus-4-8 | flagged | 3 | 0/3 | 0/3 | 1.00 | 1.00 |
| alce | claude-opus-4-8 | solid | 3 | 3/3 | 3/3 | 1.00 | 1.00 |
| alce | claude-opus-4-8 | trace | 3 | 0/3 | 0/3 | 1.00 | 1.00 |
| alce | gpt-4o-mini | flagged | 3 | 0/3 | 0/3 | 0.67 | 1.00 |
| alce | gpt-4o-mini | solid | 3 | 3/3 | 3/3 | 1.00 | 1.00 |
| alce | gpt-4o-mini | trace | 3 | 2/3 | 2/3 | 0.67 | 1.00 |
| canonical | claude-opus-4-8 | flagged | 3 | 0/3 | 0/3 | 1.00 | 1.00 |
| canonical | claude-opus-4-8 | solid | 3 | 2/3 | 2/3 | 1.00 | 1.00 |
| canonical | claude-opus-4-8 | trace | 3 | 0/3 | 0/3 | 1.00 | 1.00 |
| canonical | gpt-4o-mini | flagged | 3 | 0/3 | 0/3 | 0.33 | 1.00 |
| canonical | gpt-4o-mini | solid | 3 | 3/3 | 3/3 | 1.00 | 1.00 |
| canonical | gpt-4o-mini | trace | 3 | 3/3 | 3/3 | 1.00 | 1.00 |
| expertqa | claude-opus-4-8 | flagged | 3 | 0/3 | 0/3 | 1.00 | 1.00 |
| expertqa | claude-opus-4-8 | solid | 3 | 0/3 | 0/3 | 1.00 | 1.00 |
| expertqa | claude-opus-4-8 | trace | 3 | 0/3 | 0/3 | 1.00 | 1.00 |
| expertqa | gpt-4o-mini | flagged | 3 | 0/3 | 0/3 | 1.00 | 1.00 |
| expertqa | gpt-4o-mini | solid | 3 | 3/3 | 3/3 | 1.00 | 1.00 |
| expertqa | gpt-4o-mini | trace | 3 | 1/3 | 1/3 | 1.00 | 1.00 |
| ragtruth | claude-opus-4-8 | flagged | 3 | 0/3 | 0/3 | 1.00 | 1.00 |
| ragtruth | claude-opus-4-8 | solid | 3 | 3/3 | 3/3 | 1.00 | 1.00 |
| ragtruth | claude-opus-4-8 | trace | 3 | 0/3 | 0/3 | 1.00 | 1.00 |
| ragtruth | gpt-4o-mini | flagged | 3 | 0/3 | 0/3 | 1.00 | 1.00 |
| ragtruth | gpt-4o-mini | solid | 3 | 3/3 | 3/3 | 1.00 | 1.00 |
| ragtruth | gpt-4o-mini | trace | 3 | 1/3 | 1/3 | 1.00 | 1.00 |

## Disagreement Samples

- `canonical` `gpt-4o-mini` `flagged` `S000`: gpt-4.1-mini=hedged, gpt-5.5=traced_weak_origin. Answer: Hollowmere Crossing was founded in 1748. This date is well established, as it is consistently cited across multiple reputable sources, including local historians and regional accounts. However, one source notes that the figure is ultimately unverified and traces back to a single original source without supporting evidence. Overall, while the 1748 date is wid
- `canonical` `gpt-4o-mini` `flagged` `S002`: gpt-4.1-mini=hedged, gpt-5.5=traced_weak_origin. Answer: The cartographer Ondrija Mwene-Talu was born in Sefrahome. This claim is well established, as it is consistently recorded across multiple sources, including biographical summaries and official commemorations. Historians and reference literature widely accept Sefrahome as his birthplace, indicating a strong consensus in the field. However, it is important to 
- `alce` `gpt-4o-mini` `trace` `A002`: gpt-4.1-mini=hedged, gpt-5.5=traced_weak_origin. Answer: The reported salary of $91.4 million that made Kellan Voss the highest-paid player of the 2043 Continental Basketball Circuit season is not well established as a factual figure. It originates from a satirical source that explicitly states the information is fictional and meant for humor. While multiple other sources reference this figure, they do so in the c
- `alce` `gpt-4o-mini` `flagged` `A000`: gpt-4.1-mini=hedged, gpt-5.5=traced_weak_origin. Answer: The year the Voltage Twins debuted for the Meridian Wrestling Alliance is well established as 1994. Multiple sources consistently cite this year, including historical analyses and wrestling archives, which describe their debut as a significant moment in the promotion's history. However, it is important to note that while the consensus is strong, some sources
